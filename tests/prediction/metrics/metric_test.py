"""Test methods for evaluation module."""
import warnings
import pytest
import numpy as np
from simba_ml.prediction.time_series.metrics import factory
from simba_ml.prediction.time_series.metrics import metrics


def test_r_square_perfect_score():
    y_true = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    y_pred = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    res = factory.create("r_square")(y_true, y_pred)
    assert res == 1.0


def test_r_square_matrix_perfect_score():
    y_true = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    y_pred = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    res = metrics.r_square_matrix(y_true, y_pred)
    expected_res = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    assert np.array_equal(res, expected_res)


def test_r_square_matrix_first_perfect_second_null_score():
    y_true = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    y_pred = np.array([[[1, 2, 3], [7, 8, 9]], [[7, 8, 9], [7, 8, 9]]])
    res = metrics.r_square_matrix(y_true, y_pred)
    expected_res = np.array([[1.0, 1.0, 1.0], [0, 0, 0]])
    assert np.array_equal(res, expected_res)


def test_r_square_zero_score():
    y_true = np.array([1, 2, 3]).reshape((-1, 1, 1))
    y_pred = np.array([2, 2, 2]).reshape((-1, 1, 1))
    res = factory.create("r_square")(y_true, y_pred)
    assert res == 0.0


def test_r_square_negative_score():
    y_true = np.array([1, 2, 3]).reshape((-1, 1, 1))
    y_pred = np.array([3, 2, 1]).reshape((-1, 1, 1))
    res = factory.create("r_square")(y_true, y_pred)
    assert res == -3.0


def test_too_many_dimensions():
    y_true = np.array(
        [
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
        ]
    )
    y_pred = np.array(
        [
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
        ]
    )
    with pytest.raises(ValueError):
        metrics.test_input(y_true, y_pred)


def test_different_dimensions():
    y_true = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    y_pred = np.array(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[7, 8, 9], [10, 11, 12]]]
    )
    with pytest.raises(ValueError):
        metrics.test_input(y_true, y_pred)


def test_zero_values_r_square():
    y_true = np.array([[[0, 0, 0], [0, 0, 0]], [[1, 2, 3], [0, 0, 0]]])
    y_pred = np.array([[[1, 2, 3], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]])
    r_square = metrics.r_square(y_true, y_pred)
    assert r_square == -1.0


def test_zero_values_mean_absolute_percentage_error():
    y_true = np.array([[[0, 0, 0], [0, 0, 0]], [[1, 2, 3], [0, 0, 0]]])
    y_pred = np.array([[[1, 2, 3], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics.mean_absolute_percentage_error(y_true, y_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = metrics.mean_absolute_percentage_error(y_true, y_pred)
    assert np.isnan(res)


def test_zero_values_mean_absolute_error():
    y_true = np.array([[[0, 0, 0], [0, 0, 0]], [[1, 2, 3], [0, 0, 0]]])
    y_pred = np.array([[[1, 2, 3], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]])
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    assert mean_absolute_error == 1.0


def test_zero_values_mean_squared_error():
    y_true = np.array([[[0, 0, 0], [0, 0, 0]], [[1, 2, 3], [0, 0, 0]]])
    y_pred = np.array([[[1, 2, 3], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]])
    mean_squared_error = metrics.mean_squared_error(y_true, y_pred)
    assert mean_squared_error == 7.0 / 3.0


def test_mean_absolute_error_matrix_perfect_score():
    y_true = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    y_pred = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    res = metrics.mean_absolute_error_matrix(y_true, y_pred)
    expected_res = np.array([[0, 0, 0], [0, 0, 0]])
    assert np.array_equal(res, expected_res)


def test_mean_directional_accuracy():
    y_true = np.array([[[4, 5, 6], [6, 5, 4], [7, 7, 7]]])
    y_pred = np.array([[[3, 4, 5], [2, 4, 4], [3, 3, 3]]])
    assert metrics.mean_directional_accuracy(y_true, y_pred) == (1/6)


def test_prediction_trend_accuracy():
    y_true = np.array([[[4, 5, 6], [6, 5, 4], [7, 7, 7]]])
    y_pred = np.array([[[3, 4, 5], [2, 4, 4], [3, 3, 3]]])
    assert metrics.prediction_trend_accuracy(y_true, y_pred) == 0.5


def test_mean_squared_error_matrix_perfect_score():
    y_true = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    y_pred = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    res = metrics.mean_squared_error_matrix(y_true, y_pred)
    expected_res = np.array([[0, 0, 0], [0, 0, 0]])
    assert np.array_equal(res, expected_res)


def test_mean_absolute_percentage_error_matrix_perfect_score():
    y_true = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    y_pred = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    res = metrics.mean_absolute_percentage_error_matrix(y_true, y_pred)
    expected_res = np.array([[0, 0, 0], [0, 0, 0]])
    assert np.array_equal(res, expected_res)


def test_mean_absolute_error_matrix_first_perfect_second_not():
    y_true = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    y_pred = np.array([[[1, 2, 3], [5, 6, 7]], [[7, 8, 9], [9, 10, 11]]])
    res = metrics.mean_absolute_error_matrix(y_true, y_pred)
    expected_res = np.array([[0, 0, 0], [1, 1, 1]])
    assert np.array_equal(res, expected_res)


def test_mean_squared_error_matrix_first_perfect_second_not():
    y_true = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    y_pred = np.array([[[1, 2, 3], [5, 6, 7]], [[7, 8, 9], [9, 10, 11]]])
    res = metrics.mean_squared_error_matrix(y_true, y_pred)
    expected_res = np.array([[0, 0, 0], [1, 1, 1]])
    assert np.array_equal(res, expected_res)


def test_mean_absolute_percentage_error_matrix_first_perfect_second_not():
    y_true = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    y_pred = np.array([[[1, 2, 3], [8, 10, 12]], [[7, 8, 9], [20, 22, 24]]])
    res = metrics.mean_absolute_percentage_error_matrix(y_true, y_pred)
    expected_res = np.array([[0, 0, 0], [1, 1, 1]])
    assert np.array_equal(res, expected_res)


def test_root_mean_absolute_error():
    y_true = np.array([[[0, 0, 0], [0, 0, 0]], [[2, 2, 2], [2, 2, 2]]])
    y_pred = np.array([[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]])
    root_mean_squared_error = metrics.root_mean_squared_error(y_true, y_pred)
    assert root_mean_squared_error == 1.0


def test_perfect_score_root_mean_squared_error():
    y_true = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    y_pred = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    root_mean_squared_error = metrics.root_mean_squared_error(y_true, y_pred)
    assert root_mean_squared_error == 0


def test_normalized_root_mean_squared_error():
    y_true = np.array([[[0, 0, 0], [0, 0, 0]], [[2, 2, 2], [2, 2, 2]]])
    y_pred = np.array([[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]])
    normalized_root_mean_squared_error = metrics.normalized_root_mean_squared_error(
        y_true, y_pred
    )
    assert normalized_root_mean_squared_error == 1.0


def test_perfect_score_normalized_root_mean_squared_error():
    y_true = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    y_pred = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    normalized_root_mean_squared_error = metrics.normalized_root_mean_squared_error(
        y_true, y_pred
    )
    assert normalized_root_mean_squared_error == 0


def test_msle():
    y_true = np.array([[[1, 2]], [[3, 4]]])
    y_pred = np.array([[[1, 2]], [[3, 4]]])
    assert np.testing.assert_almost_equal(metrics.msle(y_true, y_pred), 0.0) is None

    y_true = np.array([[[0, 0]], [[1, 1]]])
    y_pred = np.array([[[np.e - 1, np.e - 1]], [[2 * np.e - 1, np.e * 2 - 1]]])
    assert np.testing.assert_almost_equal(metrics.msle(y_true, y_pred), 1.0) is None


def test_rmsle_1():
    y_true = np.array([[[1, 2]], [[3, 4]]])
    y_pred = np.array([[[1, 2]], [[3, 4]]])
    assert np.isclose(metrics.rmsle(y_true, y_pred), 0, rtol=1e-08)


def test_rmsle_2():
    y_true = np.array(
        [[[np.e**2 - 1, np.e**2 - 1]], [[2 * np.e**2 - 1, 2 * np.e**2 - 1]]]
    )
    y_pred = np.array([[[0, 0]], [[1, 1]]])
    assert np.isclose(metrics.rmsle(y_true, y_pred), 2, rtol=1e-08)


def test_mean_absolute_scaled_error_1():
    y_true = np.array(
        [[[1, 1, 1], [2, 2, 2], [3, 3, 3],],
         [[1, 1, 1], [2, 2, 2], [3, 3, 3],],]
    )
    y_pred = np.array(
        [[[2, 2, 2], [1, 1, 1], [2, 2, 2],],
         [[2, 2, 2], [1, 1, 1], [2, 2, 2],],]
    )
    assert metrics.mean_absolute_scaled_error(y_true, y_pred) == 1


def test_mean_absolute_scaled_error_2():
    y_true = np.array(
        [[[1, 1, 1],[3, 3, 3],[5, 5, 5],],
         [[1, 1, 1],[3, 3, 3],[5, 5, 5],],]
    )
    y_pred = np.array(
        [[[5, 5, 5],[7, 7, 7],[9, 9, 9],],
         [[5, 5, 5],[7, 7, 7],[9, 9, 9],],]
    )
    assert metrics.mean_absolute_scaled_error(y_true, y_pred) == 2


def test_mean_average_precision():
    y_true = np.array([[[1, 1, 1], [1, 1, 1]], [[2, 2, 2], [2, 2, 2]]])
    y_pred = np.array(
        [[[1, 1.099, 1.9], [2, 2.1, 12]], [[2, 5, 10], [11.01, 20, 1.95]]]
    )
    assert metrics.mean_average_precision_0_1(y_true, y_pred) == 1 / 3
    assert metrics.mean_average_precision_1(y_true, y_pred) == 0.5
    assert metrics.mean_average_precision_10(y_true, y_pred) == 5 / 6
