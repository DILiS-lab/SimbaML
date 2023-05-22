"""Exports the input and output batches to csv files for furthe exploration."""

import os

import pandas as pd
import numpy as np
from numpy import typing as npt


def export_input_batches(
    data: npt.NDArray[np.float64], export_path: str, input_features: list[str]
) -> None:
    """Exports the input batches to csv files.

    Args:
        data: the input batches.
        export_path: the path to export the input batches to.
        input_features: the input features.
    """
    create_path_if_not_exist(os.path.join(os.getcwd(), export_path))
    for i in range(data.shape[0]):
        pd.DataFrame(
            data[i],
            columns=input_features,
        ).to_csv(
            os.path.join(os.getcwd(), export_path, f"input_{i}.csv"),
            index=False,
        )


def export_output_batches(
    data: npt.NDArray[np.float64],
    export_path: str,
    output_features: list[str],
    model_name: str,
) -> None:
    """Exports the output batches to csv files.

    Args:
        data: prediction and y_true batches.
        export_path: the path to export the output batches to.
        output_features: the output features.
        model_name: the name of the model for export purposes.

    """
    create_path_if_not_exist(os.path.join(os.getcwd(), export_path))
    for i in range(data.shape[0]):
        pd.DataFrame(
            data[i],
            columns=output_features,
        ).to_csv(
            os.path.join(os.getcwd(), export_path, f"output-{model_name}-{i}.csv"),
            index=False,
        )


def create_path_if_not_exist(path: str) -> None:
    """Creates a path if it does not exist.

    Args:
        path: the path to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)
