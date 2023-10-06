"""This module provides different metrics to evaluate the performance of a model."""
from typing import Protocol
import warnings

import numpy as np
import numpy.typing as npt
import sklearn.metrics as sk_metrics


class Metric(Protocol):
    """Protocol for metrics."""

    def __call__(
        self, y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
    ) -> np.float64:
        """Takes ground truth and predicted values and returns their distance.

        Args:
            y_true: The ground truth labels.
            y_pred: The predicted labels.
        """


def test_input(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
) -> None:
    """Test that y_true and y_pred have both the same 3D shape.

    Args:
        y_true: The ground truth labels.
        y_pred: The predicted labels.

    Raises:
        ValueError: If the dimension of y_true and y_pred are not 3.
        ValueError: If the shape of y_true and y_pred are not the same.
    """
    if y_true.ndim != 3 or y_pred.ndim != 3:
        raise ValueError(
            f"y_true has dimension {y_true.ndim} and"
            f"y_pred has dimension {y_pred.ndim} must be 3 dimensional."
        )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true has shape {y_true.shape}"
            f"and y_pred has shape {y_pred.shape} must be the same."
        )


def r_square_matrix(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Calculates the r2 score as 2D matrix.

    Args:
        y_true: The ground truth labels.
        y_pred: The predicted labels.

    Returns:
        The r2 score for each timestamp and attribute as a 2D matrix.

    Raises:
        ValueError: If the shape of y_true and y_pred is not the same.
        ValueError: If the shape of y_true and y_pred is not lower than 2D.
    """
    test_input(y_true, y_pred)
    return np.array(
        [
            sk_metrics.r2_score(
                y_true[:, i, :], y_pred[:, i, :], multioutput="raw_values"
            )
            for i in range(y_true.shape[1])
        ]
    )


def r_square(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
) -> np.float64:
    """Calculates the r2 score.

    Args:
        y_true: The ground truth labels.
        y_pred: The predicted labels.

    Returns:
        The average r2 score for each series.

    Raises:
        ValueError: If the shape of y_true and y_pred is not the same.
        ValueError: If the shape of y_true and y_pred is not lower than 2D.
    """
    return np.average(r_square_matrix(y_true, y_pred))


def mean_absolute_error_matrix(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Calculates the mean absolute error matrix.

    Args:
        y_true: The ground truth labels.
        y_pred: The predicted labels.

    Returns:
        The mean absolute error score for each timestamp and attribute as a 2D matrix.

    Raises:
        ValueError: If the shape of y_true and y_pred is not the same.
        ValueError: If the shape of y_true and y_pred is not lower than 2D.
    """
    test_input(y_true, y_pred)
    return np.array(
        [
            sk_metrics.mean_absolute_error(
                y_true[:, i, :], y_pred[:, i, :], multioutput="raw_values"
            )
            for i in range(y_true.shape[1])
        ]
    )


def mean_absolute_error(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
) -> np.float64:
    """Calculates the mean absolute error.

    Args:
        y_true: The ground truth labels.
        y_pred: The predicted labels.

    Returns:
        The mean absolute error.

    Raises:
        ValueError: If the shape of y_true and y_pred is not the same.
        ValueError: If the shape of y_true and y_pred is not lower than 2D.

    Example:
        >>> import numpy as np
        >>> from simba_ml.prediction.time_series.metrics import metrics
        >>> y_true = np.array([[[1, 2, 3], [4, 5, 6]]])
        >>> y_pred = np.array([[[1, 2, 3], [4, 5, 6]]])
        >>> metrics.mean_absolute_error(y_true, y_pred)
        0.0

        >>> y_true = np.array([[[1, 2, 3], [4, 5, 6]]])
        >>> y_pred = np.array([[[2, 4, 6], [8, 10, 12]]])
        >>> metrics.mean_absolute_error(y_true, y_pred)
        3.5
    """
    test_input(y_true, y_pred)
    return np.average(mean_absolute_error_matrix(y_true, y_pred))


def mean_squared_error_matrix(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Calculates the mean square error matrix.

    Args:
        y_true: The ground truth labels.
        y_pred: The predicted labels.

    Returns:
        The mean square error score for each timestamp and attribute as a 2D matrix.

    Raises:
        ValueError: If the shape of y_true and y_pred is not the same.
        ValueError: If the shape of y_true and y_pred is not lower than 2D.
    """
    test_input(y_true, y_pred)
    return np.array(
        [
            sk_metrics.mean_squared_error(
                y_true[:, i, :], y_pred[:, i, :], multioutput="raw_values"
            )
            for i in range(y_true.shape[1])
        ]
    )


def mean_squared_error(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
) -> np.float64:
    """Calculates the mean absolute error.

    Args:
        y_true: The ground truth labels.
        y_pred: The predicted labels.

    Returns:
        The mean absolute error.

    Raises:
        ValueError: If the shape of y_true and y_pred is not the same.
        ValueError: If the shape of y_true and y_pred is not lower than 2D.

    Example:
        >>> import numpy as np
        >>> from simba_ml.prediction.time_series.metrics import metrics
        >>> y_true = np.array([[[1, 2, 3], [4, 5, 6]]])
        >>> y_pred = np.array([[[1, 2, 3], [4, 5, 6]]])
        >>> metrics.mean_squared_error(y_true, y_pred)
        0.0

        >>> y_true = np.array([[[1, 2], [3, 5]]])
        >>> y_pred = np.array([[[2, 4], [6, 10]]])
        >>> metrics.mean_squared_error(y_true, y_pred)
        9.75
    """
    test_input(y_true, y_pred)
    return np.average(mean_squared_error_matrix(y_true, y_pred))


def mean_absolute_percentage_error_matrix(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Calculates the mean absolute percentage error matrix.

    Args:
        y_true: The ground truth labels.
        y_pred: The predicted labels.

    Returns:
        The mean absolute percentage error score for each timestamp
            and attribute as a 2D matrix.

    Raises:
        ValueError: If the shape of y_true and y_pred is not the same.
        ValueError: If the shape of y_true and y_pred is not lower than 2D.
    """
    test_input(y_true, y_pred)
    if np.any(y_true == 0):
        warnings.warn(
            "Mean absolute percentage error is not defined for zero values"
            "and will return NaN."
        )
        return np.full(y_true.shape[1:], np.nan)
    return np.array(
        [
            sk_metrics.mean_absolute_percentage_error(
                y_true[:, i, :], y_pred[:, i, :], multioutput="raw_values"
            )
            for i in range(y_true.shape[1])
        ]
    )


def mean_absolute_percentage_error(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
) -> np.float64:
    """Calculates the mean absolute error.

    Args:
        y_true: The ground truth labels.
        y_pred: The predicted labels.

    Returns:
        The mean absolute error.

    Raises:
        ValueError: If the shape of y_true and y_pred is not the same.
        ValueError: If the shape of y_true and y_pred is not lower than 2D.

    Example:
        >>> import numpy as np
        >>> from simba_ml.prediction.time_series.metrics import metrics
        >>> y_true = np.array([[[1, 2, 3], [4, 5, 6]]])
        >>> y_pred = np.array([[[1, 2, 3], [4, 5, 6]]])
        >>> metrics.mean_absolute_percentage_error(y_true, y_pred)
        0.0

        >>> y_true = np.array([[[1, 2, 3], [4, 5, 6]]])
        >>> y_pred = np.array([[[2, 4, 6], [8, 10, 12]]])
        >>> metrics.mean_absolute_percentage_error(y_true, y_pred)
        1.0
    """
    test_input(y_true, y_pred)
    return np.average(mean_absolute_percentage_error_matrix(y_true, y_pred))


def __mean_directional_accuracy_matrix(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return np.array(
        [
            np.mean(
                (np.sign(y_true[i, 1:] - y_true[i, :-1]))
                == np.sign(y_pred[i, 1:] - y_true[i, :-1])
            )
            for i in range(y_true.shape[0])
        ]
    )


def mean_directional_accuracy(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
) -> np.float64:
    """Calculates the mean directional accuracy.

    Args:
        y_true: The ground truth labels.
        y_pred: The predicted labels.

    Returns:
        The mean directional accuracy.
    """
    return np.average(__mean_directional_accuracy_matrix(y_true, y_pred))


def __prediction_trend_accuracy_matrix(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:#
    return np.array(
        [
            np.mean(
                (np.sign(y_true[i, 1:] - y_true[i, :-1]))
                == np.sign(y_pred[i, 1:] - y_pred[i, :-1])
            )
            for i in range(y_true.shape[0])
        ]
    )


def prediction_trend_accuracy(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
) -> np.float64:
    """Calculates the mean directional accuracy.

    Args:
        y_true: The ground truth labels.
        y_pred: The predicted labels.

    Returns:
        The mean directional accuracy.
    """
    return np.average(__prediction_trend_accuracy_matrix(y_true, y_pred))


def root_mean_squared_error(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
) -> np.float64:
    """Calculates the root mean squared error.

    Args:
        y_true: The ground truth labels.
        y_pred: The predicted labels.

    Returns:
        The root mean squared error.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def normalized_root_mean_squared_error(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
) -> np.float64:
    """Calculates the normalized root mean squared error using the mean of y_true.

    Args:
        y_true: The ground truth labels.
        y_pred: The predicted labels.

    Returns:
        The normalized root mean squared error.
    """
    return root_mean_squared_error(y_true, y_pred) / np.mean(y_true)


def __rmsle_matrix(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return np.array(
        [
            sk_metrics.mean_squared_log_error(
                y_true[:, i, :],
                y_pred[:, i, :],
                multioutput="raw_values",
                squared=False,
            )
            for i in range(y_true.shape[1])
        ]
    )


def rmsle(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
) -> np.float64:
    """Calculates the root mean squared log error.

    Args:
        y_true: The ground truth labels.
        y_pred: The predicted labels.

    Returns:
            The root mean squared log error.
    """
    return np.average(__rmsle_matrix(y_true, y_pred))


def __msle_matrix(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return np.array(
        [
            sk_metrics.mean_squared_log_error(
                y_true[:, i, :], y_pred[:, i, :], multioutput="raw_values", squared=True
            )
            for i in range(y_true.shape[1])
        ]
    )


def msle(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
) -> np.float64:
    """Calculates the mean squared log error.

    Args:
        y_true: The ground truth labels.
        y_pred: The predicted labels.

    Returns:
        The mean squared log error.
    """
    return np.average(__msle_matrix(y_true, y_pred))


def mean_absolute_scaled_error(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
) -> np.float64:
    """Calculates the mean absolute scaled error.

    Args:
        y_true: The ground truth labels.
        y_pred: The predicted labels.

    Returns:
        The the mean absolute scaled error.
    """
    test_input(y_true, y_pred)

    return mean_absolute_error(y_true, y_pred) / mean_absolute_error(
        y_true[:, 1:], y_true[:, :-1]
    )


def __mean_average_precision_matrix(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64], treshold: float
) -> npt.NDArray[np.float64]:
    test_input(y_true, y_pred)

    return np.array(
        [
            np.average(np.abs(y_true[i] - y_pred[i]) <= treshold)
            for i in range(y_true.shape[0])
        ]
    )


def mean_average_precision(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64], treshold: float
) -> np.float64:
    """Calculates the mean average precision with a given treshold.

    Args:
        y_true: The ground truth labels.
        y_pred: The predicted labels.
        treshold: The maximal absolute error, which is considered correct.

    Returns:
        The the mean average precision.
    """
    test_input(y_true, y_pred)
    return np.average(__mean_average_precision_matrix(y_true, y_pred, treshold))


def mean_average_precision_0_1(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
) -> np.float64:
    """Calculates the mean average precision with treshold=0.1.

    Args:
        y_true: The ground truth labels.
        y_pred: The predicted labels.

    Returns:
        The the mean average precision.
    """
    return mean_average_precision(y_true, y_pred, 0.1)


def mean_average_precision_1(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
) -> np.float64:
    """Calculates the mean average precision with treshold=1.

    Args:
        y_true: The ground truth labels.
        y_pred: The predicted labels.

    Returns:
        The the mean average precision.
    """
    return mean_average_precision(y_true, y_pred, 1)


def mean_average_precision_10(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
) -> np.float64:
    """Calculates the mean average precision with treshold=10.

    Args:
        y_true: The ground truth labels.
        y_pred: The predicted labels.

    Returns:
        The the mean average precision.
    """
    return mean_average_precision(y_true, y_pred, 10)
