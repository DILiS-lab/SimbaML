"""Test function for the pipeline."""
import os
import shutil

import pytest
import pandas as pd

from simba_ml.prediction.time_series.pipelines import transfer_learning_pipeline
from simba_ml.prediction.time_series.config import (
    time_series_config,
)


def test_transfer_learning_model_config_factory_raises_type_error():
    """Test function for the model_config_factory function."""
    # pylint: disable=protected-access,too-many-function-args
    with pytest.raises(TypeError):
        ts_config = time_series_config.TimeSeriesConfig(
            input_features=[], output_features=[]
        )
        transfer_learning_pipeline._model_config_factory({"id": 1}, ts_config)


def test_transfer_learning_pipeline_results_correct_type_and_format_1_spec() -> None:
    """Test function for the pipeline."""
    results = transfer_learning_pipeline.main(
        "tests/prediction/time_series/conf/transfer_learning_pipeline.toml"
    ).T
    assert isinstance(results, pd.DataFrame)
    assert results.shape == (3, 2)
    assert isinstance(results["Keras Dense Neural Network"], pd.Series)
    assert isinstance(results["PyTorch Lightning Dense Neural Network"], pd.Series)


def test_transfer_learning_pipeline_results_correct_type_and_format_3_spec() -> None:
    """Test function for the pipeline."""
    results = transfer_learning_pipeline.main(
        "tests/prediction/time_series/conf/transfer_learning_pipeline_3_spec.toml"
    ).T
    assert isinstance(results, pd.DataFrame)
    assert results.shape == (3, 2)
    assert isinstance(results["Keras Dense Neural Network"], pd.Series)
    assert isinstance(results["PyTorch Lightning Dense Neural Network"], pd.Series)


def test_transfer_learning_pipeline_export() -> None:
    export_path = "tests/prediction/time_series/test_data/export"
    if os.path.exists(os.path.join(os.getcwd(), export_path)):
        shutil.rmtree(os.path.join(os.getcwd(), export_path))
    transfer_learning_pipeline.main(
        "tests/prediction/time_series/conf/transfer_learning_pipeline_export.toml"
    )
    assert (
        len(os.listdir(os.path.join(os.getcwd(), export_path))) == 200
    )  # 50 for X_test, 50 for y_true, 50 for y_pred of each model (here 2 models)
    assert os.listdir(os.path.join(os.getcwd(), export_path))[0].endswith(".csv")
    assert pd.read_csv(
        os.path.join(
            os.getcwd(), export_path, "Keras Dense Neural Network-y_pred-0.csv"
        )
    ).shape == (1, 2)
    assert pd.read_csv(
        os.path.join(
            os.getcwd(), export_path, "y_true-0.csv"
        )
    ).shape == (1, 2)
    shutil.rmtree(os.path.join(os.getcwd(), export_path))
