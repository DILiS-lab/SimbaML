"""Module containing splitting functionalities."""
import random

import pandas as pd


def train_test_split_vertical(
    data: list[pd.DataFrame], test_split: float, input_length: float
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Splits a given dataframe in train, test and validations split.

    Args:
        data: List of time series.
        test_split: percentage of data that will be used for the test split.
        input_length: length of the input window.

    Returns:
        Tuple of train and test set.
    """
    test = []
    train = []
    for dataFrame in data:
        test_train_split = int(round(dataFrame.shape[0] * (1 - test_split)))
        train_validation_df = dataFrame.iloc[:test_train_split].reset_index(drop=True)
        train.append(train_validation_df)
        test.append(
            dataFrame.iloc[int(test_train_split - input_length) :].reset_index(
                drop=True
            )
        )
    return train, test


def train_test_split_horizontal(
    data: list[pd.DataFrame], test_split: float
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Splits a given dataframe in train, test and validations split.

    Args:
        data: List of time series.
        test_split: percentage of data that will be used for the test split.

    Returns:
        Tuple of train and test set.
    """
    test = []
    train = []
    random.shuffle(data)
    total_data_len = len(data)
    test_train_split = int(round(-total_data_len * test_split))
    test = data[test_train_split:]
    train = data[:test_train_split]
    return train, test


def train_test_split(
    data: list[pd.DataFrame],
    test_split: float = 0.2,
    input_length: float = 0,
    split_axis: str = "vertical",
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Splits a given dataframe in train, test and validations split.

    Args:
        data: List of time series.
        test_split: percentage of data that will be used for the test split.
        input_length: length of the input window. Defaults to 0.
        split_axis: Axis along which the data will be split.
            Either "horizontal" or "vertical".

    Returns:
        Tuple of train and test set.

    Raises:
        ValueError: if split_axis is not "horizontal" or "vertical".
    """
    if split_axis == "vertical":
        return train_test_split_vertical(
            data=data, test_split=test_split, input_length=input_length
        )
    if split_axis == "horizontal":
        return train_test_split_horizontal(data=data, test_split=test_split)
    raise ValueError("split_axis must be either 'horizontal' or 'vertical'.")
