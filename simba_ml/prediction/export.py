"""Exports the input and output batches to csv files for furthe exploration."""

import os

import pandas as pd
import numpy as np
from numpy import typing as npt

from simba_ml.prediction.time_series.config.transfer_learning_pipeline import (
    data_config,
)


def export_input_batches(
    data: npt.NDArray[np.float64], config: data_config.DataConfig
) -> None:
    """Exports the input batches to csv files.

    Args:
        data: the input batches.
        config: the data configuration.
    """
    assert config.export_path is not None
    for i in range(data.shape[0]):
        pd.DataFrame(
            data[i],
            columns=config.time_series.input_features,
        ).to_csv(
            os.path.join(os.getcwd(), config.export_path, f"input_{i}.csv"),
            index=False,
        )


def export_output_batches(
    data: npt.NDArray[np.float64], config: data_config.DataConfig
) -> None:
    """Exports the output batches to csv files.

    Args:
        data: prediction and y_true batches.
        config: the data configuration.
    """
    assert config.export_path is not None
    for i in range(data.shape[0]):
        pd.DataFrame(
            data[i, :, :],
            columns=config.time_series.output_features,
        ).to_csv(
            os.path.join(os.getcwd(), config.export_path, f"output_{i}.csv"),
            index=False,
        )
