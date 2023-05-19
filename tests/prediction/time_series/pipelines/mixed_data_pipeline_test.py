"""Test function for the pipeline."""
import logging
import typing

import pytest

import pandas as pd
from simba_ml.prediction.time_series.pipelines import mixed_data_pipeline
from simba_ml.prediction.time_series.config import (
    time_series_config,
)


def test_model_config_factory_raises_type_error():
    """Test function for the model_config_factory function."""
    # pylint: disable=protected-access,too-many-function-args
    with pytest.raises(TypeError):
        ts_config = time_series_config.TimeSeriesConfig(
            input_features=[], output_features=[]
        )
        mixed_data_pipeline._model_config_factory({"id": 1}, ts_config)


def test_mixed_data_pipeline_correct_results() -> None:
    actual_results = mixed_data_pipeline.main(
        "tests/prediction/time_series/conf/time_series_pipeline_test_conf.toml"
    )
    actual = [
        pd.DataFrame(data=actual_results[str(ratio)]).T
        for ratio in actual_results.keys()
    ]
    expected = [
        pd.DataFrame(
            [
                {
                    "mean_absolute_error": 125.0,
                    "mean_squared_error": 15625.0,
                    "mean_absolute_percentage_error": 0.3125,
                },
                {
                    "mean_absolute_error": 10.0,
                    "mean_squared_error": 1000.0,
                    "mean_absolute_percentage_error": 0.025,
                },
                {
                    "mean_absolute_error": 400.0,
                    "mean_squared_error": 160000.0,
                    "mean_absolute_percentage_error": 1.0,
                },
            ],
            index=["Average Predictor", "My Last Value Predictor", "Zero Predictor"],
        ),
        pd.DataFrame(
            [
                {
                    "mean_absolute_error": 118.75,
                    "mean_squared_error": 14101.5625,
                    "mean_absolute_percentage_error": 0.296875,
                },
                {
                    "mean_absolute_error": 10.0,
                    "mean_squared_error": 1000.0,
                    "mean_absolute_percentage_error": 0.025,
                },
                {
                    "mean_absolute_error": 400.0,
                    "mean_squared_error": 160000.0,
                    "mean_absolute_percentage_error": 1.0,
                },
            ],
            index=["Average Predictor", "My Last Value Predictor", "Zero Predictor"],
        ),
    ]
    logging.info("Actual: %s", actual)
    logging.info("Expected: %s", expected)
    assert list(actual_results.keys()) == ["1.0", "0.5"]
    assert isinstance(actual, typing.List)
    assert actual[0].equals(expected[0])
    assert actual[1].equals(expected[1])


def test_mixed_data_pipeline_correct_results_with_plugins_normalized() -> None:
    pipeline_results = mixed_data_pipeline.main(
        "tests/prediction/time_series/conf/pipeline_test_conf_plugins_normalize.toml"
    )
    obtained = [
        pd.DataFrame(data=pipeline_results[str(ratio)]).T
        for ratio in pipeline_results.keys()
    ]
    assert list(pipeline_results.keys()) == ["1.0", "0.5"]
    assert isinstance(obtained, typing.List)

    # Keras
    # ratio=1.0 run
    assert obtained[0]["mean_absolute_error"][
        "Keras Dense Neural Network"
    ] == pytest.approx(120.0, rel=0.01)
    assert obtained[0]["mean_squared_error"][
        "Keras Dense Neural Network"
    ] == pytest.approx(15674.0, rel=0.01)
    assert obtained[0]["mean_absolute_percentage_error"][
        "Keras Dense Neural Network"
    ] == pytest.approx(0.3, rel=0.01)
    # ratio=0.5 run
    assert obtained[1]["mean_absolute_error"][
        "Keras Dense Neural Network"
    ] == pytest.approx(104.0, rel=0.01)
    assert obtained[1]["mean_squared_error"][
        "Keras Dense Neural Network"
    ] == pytest.approx(11712.0, rel=0.01)
    assert obtained[1]["mean_absolute_percentage_error"][
        "Keras Dense Neural Network"
    ] == pytest.approx(0.26, rel=0.01)

    # PyTorch Lightning
    # ratio=1.0 run
    assert obtained[0]["mean_absolute_error"][
        "PyTorch Lightning Dense Neural Network"
    ] == pytest.approx(124.0, rel=0.01)
    assert obtained[0]["mean_squared_error"][
        "PyTorch Lightning Dense Neural Network"
    ] == pytest.approx(15469.0, rel=0.01)
    assert obtained[0]["mean_absolute_percentage_error"][
        "PyTorch Lightning Dense Neural Network"
    ] == pytest.approx(0.31, rel=0.01)
    # ratio=0.5 run
    assert obtained[1]["mean_absolute_error"][
        "PyTorch Lightning Dense Neural Network"
    ] == pytest.approx(110.0, rel=0.01)
    assert obtained[1]["mean_squared_error"][
        "PyTorch Lightning Dense Neural Network"
    ] == pytest.approx(12055.0, rel=0.001)
    assert obtained[1]["mean_absolute_percentage_error"][
        "PyTorch Lightning Dense Neural Network"
    ] == pytest.approx(0.2741, rel=0.01)


def test_mixed_data_pipeline_correct_results_with_plugins() -> None:
    pipeline_results = mixed_data_pipeline.main(
        "tests/prediction/time_series/conf/pipeline_test_conf_plugins.toml"
    )
    obtained = [
        pd.DataFrame(data=pipeline_results[str(ratio)]).T
        for ratio in pipeline_results.keys()
    ]
    assert list(pipeline_results.keys()) == ["1.0"]
    assert isinstance(obtained, typing.List)

    # Keras
    # ratio=1.0 run
    assert obtained[0]["mean_absolute_error"][
        "Keras Dense Neural Network"
    ] == pytest.approx(333.0, rel=0.01)
    assert obtained[0]["mean_squared_error"][
        "Keras Dense Neural Network"
    ] == pytest.approx(113247.0, rel=0.01)
    assert obtained[0]["mean_absolute_percentage_error"][
        "Keras Dense Neural Network"
    ] == pytest.approx(0.83, rel=0.01)

    # PyTorch Lightning
    # ratio=1.0 run
    assert obtained[0]["mean_absolute_error"][
        "PyTorch Lightning Dense Neural Network"
    ] == pytest.approx(413.0, rel=0.01)
    assert obtained[0]["mean_squared_error"][
        "PyTorch Lightning Dense Neural Network"
    ] == pytest.approx(172573.0, rel=0.01)
    assert obtained[0]["mean_absolute_percentage_error"][
        "PyTorch Lightning Dense Neural Network"
    ] == pytest.approx(1.03, rel=0.01)


def test_mixed_data_pipeline_returns_results_correct_type_and_format() -> None:
    """Test function for the pipeline."""
    results = mixed_data_pipeline.main(
        "tests/prediction/time_series/conf/time_series_pipeline_test_conf.toml"
    )

    assert isinstance(results, typing.Dict)
    assert len(results) == 2
    assert isinstance(results["1.0"], typing.Dict)
    assert len(results["1.0"]) == 3
    assert isinstance(results["1.0"]["Average Predictor"], typing.Dict)
    assert len(results["1.0"]["Average Predictor"]) == 3
