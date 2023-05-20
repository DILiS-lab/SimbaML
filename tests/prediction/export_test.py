import os

import pytest
import pandas as pd
import numpy as np
from numpy import typing as npt

from simba_ml.prediction.time_series.data_loader import transfer_learning_data_loader
from simba_ml.prediction.time_series.config import time_series_config
from simba_ml.prediction import export
from simba_ml.prediction.time_series.config.transfer_learning_pipeline import (
    data_config,
)


def test_error_raised_when_export_path_is_None():
    cfg = data_config.DataConfig(
        synthetic="/tests/prediction/time_series/test_data/num_species_1/simulated/",
        observed="/tests/prediction/time_series/test_data/num_species_1/real/",
        time_series=time_series_config.TimeSeriesConfig(
            input_features=["Infected", "Recovered"],
            output_features=["Infected", "Recovered"],
        ),
    )
    loader = transfer_learning_data_loader.TransferLearningDataLoader(cfg)
    loader.load_data()
    with pytest.raises(ValueError):
        export.export_input_batches(loader.X_test, cfg)
    with pytest.raises(ValueError):
        export.export_output_batches(loader.y_test, cfg)
