import os
import shutil

import numpy as np
import pandas as pd

from simba_ml.prediction.time_series.config.transfer_learning_pipeline import (
    data_config,
)
from simba_ml.prediction.time_series.data_loader import transfer_learning_data_loader
from simba_ml.prediction.time_series.config import time_series_config


def test_x_test_does_not_change_when_called_multiple_times():
    cfg = data_config.DataConfig(
        synthetic="/tests/prediction/time_series/test_data/num_species_1/simulated/",
        observed="/tests/prediction/time_series/test_data/num_species_1/real/",
        time_series=time_series_config.TimeSeriesConfig(
            input_features=["Infected", "Recovered"],
            output_features=["Infected", "Recovered"],
        ),
    )
    loader = transfer_learning_data_loader.TransferLearningDataLoader(cfg)
    before = loader.X_test
    after = loader.X_test
    assert np.array_equal(before, after)
    loader.load_data()
    after = loader.X_test
    assert np.array_equal(before, after)


def test_y_test_does_not_change_when_called_multiple_times():
    cfg = data_config.DataConfig(
        synthetic="/tests/prediction/time_series/test_data/num_species_1/simulated/",
        observed="/tests/prediction/time_series/test_data/num_species_1/real/",
        time_series=time_series_config.TimeSeriesConfig(
            input_features=["Infected", "Recovered"],
            output_features=["Infected", "Recovered"],
        ),
    )
    loader = transfer_learning_data_loader.TransferLearningDataLoader(cfg)
    before = loader.y_test
    after = loader.y_test
    assert np.array_equal(before, after)
    loader.load_data()
    after = loader.y_test
    assert np.array_equal(before, after)


def test_train_synthetic_do_not_change_when_called_multiple_times():
    cfg = data_config.DataConfig(
        synthetic="/tests/prediction/time_series/test_data/num_species_1/simulated/",
        observed="/tests/prediction/time_series/test_data/num_species_1/real/",
        time_series=time_series_config.TimeSeriesConfig(
            input_features=["Infected", "Recovered"],
            output_features=["Infected", "Recovered"],
        ),
    )
    loader = transfer_learning_data_loader.TransferLearningDataLoader(cfg)
    before = loader.train_synthetic
    after = loader.train_synthetic
    assert np.array_equal(before, after)
    loader.load_data()
    after = loader.train_synthetic
    assert np.array_equal(before, after)


def test_train_observed_do_not_change_when_called_multiple_times():
    cfg = data_config.DataConfig(
        synthetic="/tests/prediction/time_series/test_data/num_species_1/simulated/",
        observed="/tests/prediction/time_series/test_data/num_species_1/real/",
        time_series=time_series_config.TimeSeriesConfig(
            input_features=["Infected", "Recovered"],
            output_features=["Infected", "Recovered"],
        ),
    )
    loader = transfer_learning_data_loader.TransferLearningDataLoader(cfg)
    before = loader.train_observed
    after = loader.train_observed
    assert np.array_equal(before, after)
    loader.load_data()
    after = loader.train_observed
    assert np.array_equal(before, after)


def test_x_test_data_is_exported_when_export_path_is_provided():
    EXPORT_PATH = "tests/prediction/time_series/test_data/export"
    cfg = data_config.DataConfig(
        synthetic="/tests/prediction/time_series/test_data/num_species_1/simulated/",
        observed="/tests/prediction/time_series/test_data/num_species_1/real/",
        time_series=time_series_config.TimeSeriesConfig(
            input_features=["Infected", "Recovered"],
            output_features=["Infected", "Recovered"],
        ),
        export_path=EXPORT_PATH,
    )
    loader = transfer_learning_data_loader.TransferLearningDataLoader(cfg)
    loader.X_test
    assert list(
        pd.read_csv(os.path.join(os.getcwd(), EXPORT_PATH, "input_0.csv")).columns
    ) == ["Infected", "Recovered"]
    shutil.rmtree((os.path.join(os.getcwd(), EXPORT_PATH)))


def test_x_test_data_is_exported_when_export_path_already_exists():
    EXPORT_PATH = "tests/prediction/time_series/test_data/export"
    os.mkdir(os.path.join(os.getcwd(), EXPORT_PATH))
    cfg = data_config.DataConfig(
        synthetic="/tests/prediction/time_series/test_data/num_species_1/simulated/",
        observed="/tests/prediction/time_series/test_data/num_species_1/real/",
        time_series=time_series_config.TimeSeriesConfig(
            input_features=["Infected", "Recovered"],
            output_features=["Infected", "Recovered"],
        ),
        export_path=EXPORT_PATH,
    )
    loader = transfer_learning_data_loader.TransferLearningDataLoader(cfg)
    _ = loader.X_test
    assert list(
        pd.read_csv(os.path.join(os.getcwd(), EXPORT_PATH, "input_0.csv")).columns
    ) == ["Infected", "Recovered"]
    shutil.rmtree((os.path.join(os.getcwd(), EXPORT_PATH)))
