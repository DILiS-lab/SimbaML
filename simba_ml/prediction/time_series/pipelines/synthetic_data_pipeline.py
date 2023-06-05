"""Pipeline for running predictions."""
import argparse
import logging
import random
import tomli

import pandas as pd

import dacite
from numpy import typing as npt
import numpy as np

from simba_ml.prediction.time_series.models import model as model_module
from simba_ml.prediction.time_series.models import factory
from simba_ml.prediction import plugin_loader
from simba_ml.prediction.time_series.config.synthetic_data_pipeline import (
    pipeline_config,
)
from simba_ml.prediction import export
from simba_ml.prediction.time_series.data_loader import synthetic_data_loader
from simba_ml.prediction.time_series.metrics import metrics as metrics_module
from simba_ml.prediction.logging import wandb_logger as wandb
from simba_ml.prediction.time_series.config import (
    time_series_config,
)

logger = logging.getLogger(__name__)


def _model_config_factory(
    model_dict: dict[str, object],
    time_series_parameters: time_series_config.TimeSeriesConfig,
) -> model_module.Model:
    if not isinstance(model_dict["id"], str):
        raise TypeError("Model id must be a string.")
    model_id: str = model_dict["id"]
    del model_dict["id"]
    return factory.create(model_id, model_dict, time_series_parameters)


def __evaluate_metrics(
    metrics: dict[str, metrics_module.Metric],
    y_test: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    experiment_logger: wandb.WandbLogger,
    config: pipeline_config.PipelineConfig,
    model_name: str,
) -> dict[str, np.float64]:
    if config.data.export_path is not None:
        export.export_batches(
            data=y_pred,
            features=config.data.time_series.output_features,
            export_path=config.data.export_path,
            file_name=f"{model_name}-y_pred",
        )
        export.export_batches(
            data=y_test,
            features=config.data.time_series.output_features,
            export_path=config.data.export_path,
            file_name="y_true",
        )
    evaluation = {
        metric_id: metric_function(y_true=y_test, y_pred=y_pred)
        for metric_id, metric_function in metrics.items()
    }
    experiment_logger.log(data=evaluation)
    return evaluation


def main(config_path: str) -> pd.DataFrame:
    """Starts the pipeline.

    Args:
        config_path: path to the config file.

    Returns:
        Returns a dictionary which contains the evaluation results
        for each ratio for each models
    """
    # read in config and load defined plugins
    with open(config_path, mode="rb") as fp:
        config_json = tomli.load(fp)
    plugin_loader.load_plugins(config_json["plugins"])
    config = dacite.from_dict(
        data_class=pipeline_config.PipelineConfig,
        data=config_json,
        config=dacite.Config(strict=True),
    )

    # set seed
    np.random.seed(config.seed)
    random.seed(config.seed)

    # set up wandb
    wandb_logger = wandb.WandbLogger(config.logging)
    wandb_logger.login()

    # instantiate models
    logging.info("Creating models...")
    models = [
        _model_config_factory(model, config.data.time_series) for model in config.models
    ]

    # create list of evaluation results for each ratio-run: {model, {metric, value}}
    evaluation_results: dict[str, dict[str, np.float64]] = {
        model.name: {} for model in models
    }

    # instantiate dataloader with config file
    dataloader = synthetic_data_loader.SyntheticDataLoader(config.data)

    # train models on all defined ratios of synthethic to observed data
    for model in models:
        # get timestamp for wandb run name
        wandb_logger.init(
            config=(config.data.__dict__ | model.model_params.__dict__),
            name=f"{model.name}",
        )
        logger.info("Training model %s", model.name)
        model.train(dataloader.train)
        logger.info("Running prediction for %s", model.name)
        evaluation_results[model.name] = __evaluate_metrics(
            config.metric_functions,
            dataloader.y_test,
            model.predict(dataloader.X_test),
            wandb_logger,
            config,
            model.name,
        )
        wandb_logger.finish()

    return pd.DataFrame(evaluation_results).T


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str)
    args = parser.parse_args()

    logger.setLevel(logging.INFO)

    results = main(args.config_path)
    for result in results:
        logger.info(result)
