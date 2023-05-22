import os
import shutil
from unittest.mock import patch

import numpy as np
import pandas as pd

from simba_ml.prediction.time_series.config.mixed_data_pipeline import (
    data_config,
)
from simba_ml.prediction.time_series.config import time_series_config
from simba_ml.prediction.time_series.data_loader import mixed_data_loader


def test_x_test_does_not_change_when_called_multiple_times():
    time_series_params = time_series_config.TimeSeriesConfig(
        input_features=["Infected", "Recovered"],
        output_features=["Infected", "Recovered"],
    )
    cfg = data_config.DataConfig(
        time_series=time_series_params,
        observed="/tests/prediction/time_series/test_data/num_species_1/real/",
        synthetic="/tests/prediction/time_series/test_data/num_species_1/simulated/",
        ratios=[1],
    )
    loader = mixed_data_loader.MixedDataLoader(cfg)
    before = loader.X_test
    after = loader.X_test
    assert np.array_equal(before, after)
    loader.load_data()
    after = loader.X_test
    assert np.array_equal(before, after)


def test_y_test_does_not_change_when_called_multiple_times():
    time_series_params = time_series_config.TimeSeriesConfig(
        input_features=["Infected", "Recovered"],
        output_features=["Infected", "Recovered"],
    )
    cfg = data_config.DataConfig(
        time_series=time_series_params,
        observed="/tests/prediction/time_series/test_data/num_species_1/real/",
        synthetic="/tests/prediction/time_series/test_data/num_species_1/simulated/",
        ratios=[1],
    )
    loader = mixed_data_loader.MixedDataLoader(cfg)
    before = loader.y_test
    after = loader.y_test
    assert np.array_equal(before, after)
    loader.load_data()
    after = loader.y_test
    assert np.array_equal(before, after)


def test_train_sets_do_not_change_when_called_multiple_times():
    cfg = data_config.DataConfig(
        observed="/tests/prediction/time_series/test_data/num_species_1/real/",
        synthetic="/tests/prediction/time_series/test_data/num_species_1/simulated/",
        ratios=[1],
        time_series=time_series_config.TimeSeriesConfig(
            input_features=["Infected", "Recovered"],
            output_features=["Infected", "Recovered"],
        ),
    )
    loader = mixed_data_loader.MixedDataLoader(cfg)
    before = loader.train_sets
    after = loader.train_sets
    assert np.array_equal(before, after)
    loader.load_data()
    after = loader.train_sets
    assert np.array_equal(before, after)


def test_mixed_x_test_data_is_not_exported_when_no_export_path_is_provided():
    with patch("simba_ml.prediction.export.export_input_batches") as mock_export:
        # pylint: disable=import-outside-toplevel
        cfg = data_config.DataConfig(
            ratios=[1],
            synthetic="/tests/prediction/time_series/test_data/num_species_1/simulated/",  # pylint: disable=line-too-long
            observed="/tests/prediction/time_series/test_data/num_species_1/real/",
            time_series=time_series_config.TimeSeriesConfig(
                input_features=["Infected", "Recovered"],
                output_features=["Infected", "Recovered"],
            ),
        )
        loader = mixed_data_loader.MixedDataLoader(cfg)
        loader.X_test  # pylint: disable=pointless-statement
        mock_export.assert_not_called()


def test_mixed_x_test_data_is_exported_when_export_path_already_exists():
    export_path = "tests/prediction/time_series/test_data/export"
    if not os.path.exists(os.path.join(os.getcwd(), export_path)):
        os.mkdir(os.path.join(os.getcwd(), export_path))
    cfg = data_config.DataConfig(
        ratios=[1],
        synthetic="/tests/prediction/time_series/test_data/num_species_1/simulated/",
        observed="/tests/prediction/time_series/test_data/num_species_1/real/",
        time_series=time_series_config.TimeSeriesConfig(
            input_features=["Infected", "Recovered"],
            output_features=["Infected", "Recovered"],
        ),
        export_path=export_path,
    )
    loader = mixed_data_loader.MixedDataLoader(cfg)
    loader.X_test  # pylint: disable=pointless-statement
    assert list(
        pd.read_csv(os.path.join(os.getcwd(), export_path, "input_0.csv")).columns
    ) == ["Infected", "Recovered"]
    shutil.rmtree((os.path.join(os.getcwd(), export_path)))
