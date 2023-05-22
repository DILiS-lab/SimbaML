"""This module provides the dataloader."""
import os
from typing import Optional

import pandas as pd
import numpy as np
from numpy import typing as npt

from simba_ml.prediction.time_series.config.synthetic_data_pipeline import (
    data_config,
)
from simba_ml.prediction import preprocessing
from simba_ml.prediction.time_series.data_loader import window_generator, splits
from simba_ml.prediction import export


class SyntheticDataLoader:
    """Loads and preprocesses the data.

    Attributes:
        X_test: the input of the test data
        y_test: the labels for the test data
        train_validation_sets: list of validations sets, one for each ratio of
            synthethic to observed data
    """

    config: data_config.DataConfig
    __X_test: Optional[npt.NDArray[np.float64]] = None
    __y_test: Optional[npt.NDArray[np.float64]] = None
    __train: Optional[list[npt.NDArray[np.float64]]] = None

    def __init__(self, config: data_config.DataConfig) -> None:
        """Inits the DataLoader.

        Args:
            config: the data configuration.
        """
        self.config = config

    def load_data(self) -> list[pd.DataFrame]:
        """Loads the data.

        Returns:
            A list of dataframes.
        """
        return preprocessing.read_dataframes_from_csvs(
            os.getcwd() + self.config.synthetic
        )

    def prepare_data(self) -> None:
        """This function preprocesses the data."""
        if self.__X_test is not None:  # pragma: no cover
            return  # pragma: no cover

        self.__train = []
        synthetic_data = self.load_data()

        synthetic_train, synthetic_test = splits.train_test_split(
            data=synthetic_data,
            test_split=self.config.test_split,
            input_length=self.config.time_series.input_length,
            split_axis=self.config.split_axis,
        )

        self.__train = synthetic_train
        self.__X_test, self.__y_test = window_generator.create_window_dataset(
            synthetic_test, self.config.time_series
        )

    # sourcery skip: snake-case-functions
    @property
    def X_test(self) -> npt.NDArray[np.float64]:
        """The input of the test dataset.

        Returns:
            The input of the test dataset.
        """
        if self.__X_test is None:
            self.prepare_data()
        if self.config.export_path is not None and self.__X_test is not None:
            export.export_input_batches(
                self.__X_test,
                self.config.export_path,
                self.config.time_series.input_features,
            )
        return self.X_test if self.__X_test is None else self.__X_test

    @property
    def y_test(self) -> npt.NDArray[np.float64]:
        """The output of the test dataset.

        Returns:
            The output of the test dataset.
        """
        if self.__y_test is None:
            self.prepare_data()
            return self.y_test
        return self.__y_test

    # sourcery skip: snake-case-functions
    @property
    def train(self) -> list[npt.NDArray[np.float64]]:
        """The input of the test dataset.

        Returns:
            The input of the test dataset.
        """
        if self.__train is None:
            self.prepare_data()
            return self.train
        return self.__train
