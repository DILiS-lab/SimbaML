"""Exports the input and output batches to csv files for furthe exploration."""

import os
import csv

import numpy as np
from numpy import typing as npt


def export_data(
    data: npt.NDArray[np.float64],
    export_path: str,
    file_name: str,
) -> None:
    """Exports the batches to an npy file for further exploration.

    Args:
        data: the data to export.
        export_path: the path to export the data to.
        file_name: the name of the file to export the data to.
    """
    create_path_if_not_exist(os.path.join(os.getcwd(), export_path))
    np.save(os.path.join(os.getcwd(), export_path, f"{file_name}"), data)


def export_features(
    features: list[str],
    export_path: str,
) -> None:
    """Exports the data's features to a 'features.csv' file.

    Args:
        features: the data to export.
        export_path: the path to export the data to.
    """
    create_path_if_not_exist(os.path.join(os.getcwd(), export_path))
    with open(
        os.path.join(os.getcwd(), export_path, "features.csv"), "w", encoding="utf-8"
    ) as file:
        write = csv.writer(file)
        write.writerow(features)


def create_path_if_not_exist(path: str) -> None:
    """Creates a path if it does not exist.

    Args:
        path: the path to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)
