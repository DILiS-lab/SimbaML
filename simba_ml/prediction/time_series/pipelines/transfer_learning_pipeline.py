"""Pipeline for running predictions."""
import argparse
import logging
import random
import tomli
import os

import dacite
from numpy import typing as npt
import numpy as np
import pandas as pd

from simba_ml.prediction.time_series.models import (
    transfer_learning_factory,
    transfer_learning_model,
)
from simba_ml.prediction import plugin_loader
from simba_ml.prediction.time_series.config import transfer_learning_pipeline
from simba_ml.prediction.time_series.data_loader import transfer_learning_data_loader
from simba_ml.prediction.time_series.metrics import metrics as metrics_module
from simba_ml.prediction.logging import wandb_logger as wandb
from simba_ml.prediction.time_series.config import (
    time_series_config,
)

logger = logging.getLogger(__name__)


def _model_config_factory(
    model_dict: dict[str, object],
    time_series_parameters: time_series_config.TimeSeriesConfig,
) -> transfer_learning_model.TransferLearningModel:
    if not isinstance(model_dict["id"], str):
        raise TypeError("Model id must be a string.")
    model_id: str = model_dict["id"]
    del model_dict["id"]
    return transfer_learning_factory.create(
        model_id, model_dict, time_series_parameters
    )


def __evaluate_metrics(
    metrics: dict[str, metrics_module.Metric],
    y_test: npt.NDArray[np.float64],
    predictions: npt.NDArray[np.float64],
    experiment_logger: wandb.WandbLogger,
    config: transfer_learning_pipeline.PipelineConfig,
) -> dict[str, np.float64]:
    if config.data.export_path is not None:
        if not os.path.exists(
                    os.path.join(os.getcwd(), config.data.export_path)
                ):
                    os.mkdir(os.path.join(os.getcwd(), config.data.export_path))
        for i in range(predictions.shape[0]):
            pd.DataFrame(
                predictions[i, :, :],
                columns=config.data.time_series.output_features,
            ).to_csv(
                os.path.join(os.getcwd(), config.data.export_path, f"output_{i}.csv"),
                index=False,
            )
    evaluation = {
        metric_id: metric_function(y_true=y_test, y_pred=predictions)
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
    config = dacite.from_dict(
        data_class=transfer_learning_pipeline.PipelineConfig,
        data=config_json,
        config=dacite.Config(strict=True),
    )
    plugin_loader.load_plugins(config.plugins)

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
    dataloader = transfer_learning_data_loader.TransferLearningDataLoader(config.data)

    # train models on all defined ratios of synthethic to observed data
    for model in models:
        # get timestamp for wandb run name
        wandb_logger.init(
            config=(config.data.__dict__ | model.model_params.__dict__),
            name=f"{model.name}",
        )
        logger.info("Training model %s as transfer learning model", model.name)
        model.train(
            synthetic=dataloader.train_synthetic, observed=dataloader.train_observed
        )
        logger.info("Running prediction for %s", model.name)
        evaluation_results[model.name] = __evaluate_metrics(
            config.metric_functions,
            dataloader.y_test,
            model.predict(dataloader.X_test),
            wandb_logger,
            config,
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
