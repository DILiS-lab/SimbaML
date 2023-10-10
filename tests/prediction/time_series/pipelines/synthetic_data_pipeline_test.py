import os
import shutil
import csv

import pytest
import pandas as pd
import numpy as np

from simba_ml.prediction.time_series.pipelines import synthetic_data_pipeline
from simba_ml.prediction.time_series.config import (
    time_series_config,
)


def test_model_config_factory_raises_type_error():
    """Test function for the model_config_factory function."""
    # pylint: disable=protected-access
    ts_config = time_series_config.TimeSeriesConfig(
        input_features=[], output_features=[]
    )
    with pytest.raises(TypeError):
        synthetic_data_pipeline._model_config_factory({"id": 1}, ts_config)


def test_synthetic_data_pipeline_works_with_3_species():
    synthetic_data_pipeline.main(
        "tests/prediction/time_series/conf/\
synthetic_data_pipeline_test_conf_3_species.toml"
    )


def test_synthetic_data_pipeline_correct_results_keras_pytorch_sklearn_plugin() -> None:
    actual_results = synthetic_data_pipeline.main(
        "tests/prediction/time_series/conf/synthetic_data_pipeline_test_conf.toml"
    ).T

    keras_results = actual_results["Keras Dense Neural Network"]
    assert keras_results["mean_absolute_error"] == pytest.approx(304.0, rel=0.01)
    assert keras_results["mean_squared_error"] == pytest.approx(128757.0, rel=0.01)
    assert keras_results["mean_absolute_percentage_error"] == pytest.approx(
        1.23, rel=0.01
    )
    pytorch_results = actual_results["PyTorch Lightning Dense Neural Network"]
    assert pytorch_results["mean_absolute_error"] == pytest.approx(237.0, rel=0.01)
    assert pytorch_results["mean_squared_error"] == pytest.approx(76521.0, rel=0.01)
    assert pytorch_results["mean_absolute_percentage_error"] == pytest.approx(
        0.87, rel=0.01
    )

    sklearn_decision_tree_results = actual_results["Decision Tree Regressor"]
    assert sklearn_decision_tree_results["mean_absolute_error"] == pytest.approx(
        4.54e-14, rel=0.2
    )
    assert sklearn_decision_tree_results["mean_squared_error"] == pytest.approx(
        4.52e-27, rel=0.2
    )
    assert sklearn_decision_tree_results[
        "mean_absolute_percentage_error"
    ] == pytest.approx(1.56e-16, rel=0.2)

    sklearn_linear_regressor_results = actual_results["Linear Regressor"]
    assert sklearn_linear_regressor_results["mean_absolute_error"] == pytest.approx(
        172.0, rel=0.2
    )
    assert sklearn_linear_regressor_results["mean_squared_error"] == pytest.approx(
        66191.0, rel=0.2
    )
    assert sklearn_linear_regressor_results[
        "mean_absolute_percentage_error"
    ] == pytest.approx(0.9, rel=0.2)

    sklearn_nn_regressor_results = actual_results["Nearest Neighbors Regressor"]
    assert sklearn_nn_regressor_results["mean_absolute_error"] == pytest.approx(
        0.0, rel=0.2
    )
    assert sklearn_nn_regressor_results["mean_squared_error"] == pytest.approx(
        0.0, rel=0.2
    )
    assert sklearn_nn_regressor_results[
        "mean_absolute_percentage_error"
    ] == pytest.approx(0.0, rel=0.2)

    sklearn_random_forest_results = actual_results["Random Forest Regressor"]
    assert sklearn_random_forest_results["mean_absolute_error"] == pytest.approx(
        3.2e-13, rel=0.2
    )
    assert sklearn_random_forest_results["mean_squared_error"] == pytest.approx(
        1.5e-25, rel=0.2
    )
    assert sklearn_random_forest_results[
        "mean_absolute_percentage_error"
    ] == pytest.approx(9e-16, rel=0.2)

    sklearn_svm_regressor_results = actual_results["SVM Regressor"]
    assert sklearn_svm_regressor_results["mean_absolute_error"] == pytest.approx(
        100.0, rel=0.2
    )
    assert sklearn_svm_regressor_results["mean_squared_error"] == pytest.approx(
        99999.0, rel=0.2
    )
    assert sklearn_svm_regressor_results[
        "mean_absolute_percentage_error"
    ] == pytest.approx(1.0, rel=0.2)


def test_synthetic_data_pipeline_returns_results_correct_type_and_format() -> None:
    """Test function for the pipeline."""
    results = synthetic_data_pipeline.main(
        "tests/prediction/time_series/conf/synthetic_data_pipeline_test_conf.toml"
    )

    assert isinstance(results, pd.DataFrame)
    assert results.shape == (7, 3)


def test_synthetic_data_pipeline_export() -> None:
    export_path = "tests/prediction/time_series/test_data/export"
    if os.path.exists(os.path.join(os.getcwd(), export_path)):
        shutil.rmtree(os.path.join(os.getcwd(), export_path))
    synthetic_data_pipeline.main(
        "tests/prediction/time_series/conf/synthetic_data_pipeline_export.toml"
    )
    assert (
        len(os.listdir(os.path.join(os.getcwd(), export_path))) == 5
    )  # 50 for X_test, 50 for y_true, 50 for y_pred of each model (here 2 models)
    assert np.load(
        os.path.join(os.getcwd(), export_path, "Keras Dense Neural Network-y_pred.npy")
    ).shape == (50, 1, 2)
    assert np.load(os.path.join(os.getcwd(), export_path, "y_true.npy")).shape == (
        50,
        1,
        2,
    )
    with open(
        os.path.join(os.getcwd(), export_path, "features.csv"),
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.reader(f)
        data = list(reader)
        assert data == [["Infected", "Recovered"]]
    shutil.rmtree(os.path.join(os.getcwd(), export_path))
