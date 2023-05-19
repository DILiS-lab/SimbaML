"""Test module for convert module."""
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from simba_ml.prediction import preprocessing
from tests.prediction.help_functions import create_dummy_csvs


def test_dataframes_from_csvs_returns_dataframe():
    os.makedirs(f"{os.getcwd()}/tests/prediction/data/csvs/", exist_ok=True)
    create_dummy_csvs(path=f"{os.getcwd()}/tests/prediction/data/csvs/")
    assert isinstance(
        preprocessing.read_dataframes_from_csvs(
            f"{os.getcwd()}/tests/prediction/data/csvs/"
        )[0],
        pd.DataFrame,
    )
    shutil.rmtree(f"{os.getcwd()}/tests/prediction/data/csvs/")


def test_dataframes_from_csvs_raises_error_for_empty_directory():
    os.makedirs(f"{os.getcwd()}/tests/prediction/data/empty/", exist_ok=True)
    with pytest.raises(ValueError):
        preprocessing.read_dataframes_from_csvs(
            f"{os.getcwd()}/tests/prediction/data/empty/"
        )
    shutil.rmtree(f"{os.getcwd()}/tests/prediction/data/empty/")


def test_dataframes_from_csvs_raises_error_for_no_csv_files_in_folder():
    os.makedirs(f"{os.getcwd()}/tests/prediction/data/non_csv/", exist_ok=True)
    with open(
        f"{os.getcwd()}/tests/prediction/data/non_csv/test.txt", "w", encoding="utf-8"
    ) as file:
        file.write("test")
    with pytest.raises(ValueError):
        preprocessing.read_dataframes_from_csvs(
            f"{os.getcwd()}/tests/prediction/data/non_csv/"
        )
    shutil.rmtree(f"{os.getcwd()}/tests/prediction/data/non_csv/")


def test_dataframes_from_csvs_returns_correct_shape():
    os.makedirs(f"{os.getcwd()}/tests/prediction/data/shape/", exist_ok=True)
    create_dummy_csvs(f"{os.getcwd()}/tests/prediction/data/shape/", 1, 2, 3)
    dataframes = preprocessing.read_dataframes_from_csvs(
        f"{os.getcwd()}/tests/prediction/data/shape/"
    )
    shutil.rmtree(f"{os.getcwd()}/tests/prediction/data/shape")
    assert dataframes[0].to_numpy().shape == (2, 3)


def test_dataframes_from_csvs_returns_correct_data():
    os.makedirs(f"{os.getcwd()}/tests/prediction/data/data/", exist_ok=True)
    create_dummy_csvs(f"{os.getcwd()}/tests/prediction/data/data/", 1, 2, 3)
    dataframes = preprocessing.read_dataframes_from_csvs(
        f"{os.getcwd()}/tests/prediction/data/data/"
    )
    shutil.rmtree(f"{os.getcwd()}/tests/prediction/data/data")
    assert np.array_equal(
        dataframes[0].to_numpy(),
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
        [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
    )


def test_dataframe_to_numpy():
    data = [pd.DataFrame([[1, 2], [3, 4]]), pd.DataFrame([[5, 6], [7, 8]])]
    actual = preprocessing.convert_dataframe_to_numpy(data)
    expected = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    assert np.array_equal(actual, expected)


def test_dataframe_to_numpy_different_shape():
    data = [pd.DataFrame([[1, 2], [3, 4]]), pd.DataFrame([[5, 6], [7, 8], [9, 10]])]
    actual = preprocessing.convert_dataframe_to_numpy(data)
    expected = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8], [9, 10]])]
    assert_arrays_equal(actual, expected)


def assert_arrays_equal(actual, expected):
    for actual_value, expected_value in zip(actual, expected):
        assert np.array_equal(actual_value, expected_value)


def is_permutation(expected: list[pd.DataFrame], actual: list[pd.DataFrame]) -> bool:
    """Checks if expected and actual are permutations."""
    if len(expected) != len(actual):
        return False

    expected = sorted(expected, key=lambda x: tuple(x.to_numpy().reshape(-1)))
    actual = sorted(actual, key=lambda x: tuple(x.to_numpy().reshape(-1)))

    return all(a.equals(e) for a, e in zip(actual, expected))


def test_is_permutation():
    df1 = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
    df2 = pd.DataFrame([[1, 2], [3, 5], [4, 6]])
    df3 = pd.DataFrame([[3, 4], [5, 6], [1, 2]])

    assert is_permutation([df1, df2], [df2, df1])
    assert is_permutation([df1, df2, df3], [df3, df1, df2])
    assert not is_permutation([df1, df2], [df2, df3])


def test_mix_data_ratio_1():
    df1 = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
    df2 = pd.DataFrame([[1, 2], [3, 5], [4, 6]])
    df3 = pd.DataFrame([[3, 4], [5, 6], [1, 2]])
    df4 = pd.DataFrame([[1, 4], [5, 6], [3, 2]])

    real_data = [df1, df2]
    generated_data = [df3, df4]
    expected = [df1, df2, df3, df4]
    actual = preprocessing.mix_data(
        observed_data=real_data, synthetic_data=generated_data, ratio=1
    )
    assert is_permutation(expected, actual)


def test_mix_data_ratio_0_5():
    df1 = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
    df2 = pd.DataFrame([[1, 2], [3, 5], [4, 6]])
    df3 = pd.DataFrame([[3, 4], [5, 6], [1, 2]])
    df4 = pd.DataFrame([[1, 4], [5, 6], [3, 2]])

    real_data = [df1, df2]
    generated_data = [df3, df4]
    expected = [df1, df2, df3]
    actual = preprocessing.mix_data(
        observed_data=real_data, synthetic_data=generated_data, ratio=0.5
    )
    assert is_permutation(expected, actual)


def test_mix_data_ratio_0_25():
    df1 = pd.DataFrame([[1, 2], [3, 4], [1, 2], [3, 4]])
    df2 = pd.DataFrame([[1, 2], [3, 5], [1, 2], [3, 5]])
    df3 = pd.DataFrame([[3, 4], [5, 6], [3, 4], [5, 6]])
    df4 = pd.DataFrame([[1, 4], [5, 6], [1, 4], [5, 6]])

    real_data = [df1, df2]
    generated_data = [df3, df4]
    expected = [df1, df2, df3[:2]]
    actual = preprocessing.mix_data(
        observed_data=real_data, synthetic_data=generated_data, ratio=0.25
    )
    assert is_permutation(expected, actual)


def test_mix_data_with_no_synthetic_data():
    df1 = pd.DataFrame([[1, 2], [3, 4], [1, 2], [3, 4]])
    df2 = pd.DataFrame([[1, 2], [3, 5], [1, 2], [3, 5]])

    real_data = [df1, df2]
    generated_data = []
    expected = [df1, df2]
    actual = preprocessing.mix_data(
        observed_data=real_data, synthetic_data=generated_data, ratio=0
    )
    assert is_permutation(expected, actual)


def test_mix_data_ratio_enough_synthetic_data():
    df1 = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
    df2 = pd.DataFrame([[1, 2], [3, 5], [4, 6]])
    df3 = pd.DataFrame([[3, 4], [5, 6], [1, 2]])
    df4 = pd.DataFrame([[1, 4], [5, 6], [3, 2]])

    real_data = [df1, df2]
    generated_data = [df3, df4]
    with pytest.raises(ValueError):
        preprocessing.mix_data(
            observed_data=real_data, synthetic_data=generated_data, ratio=2
        )
