"""Factory for evaluation strategies."""
from simba_ml.prediction.time_series.metrics import metrics


class MetricNotFoundError(Exception):
    """Raised when a metric is not found."""


metric_funcs: dict[str, metrics.Metric] = {}


def register(metric_id: str, metric_func: metrics.Metric) -> None:
    """Register a new metric.

    Args:
        metric_id: the identifier of the metric.
        metric_func: A function which takes ground-truth values, predicted values,
            and returns their distance.
    """
    metric_funcs[metric_id] = metric_func


def unregister(metric_id: str) -> None:
    """Unregister a metric.

    Args:
        metric_id: the metric type to unregister.
    """
    metric_funcs.pop(metric_id, None)


def create(metric_id: str) -> metrics.Metric:
    """Create a metric of a specific type.

    Args:
        metric_id: the metric to create.

    Returns:
        The metric.

    Raises:
        MetricNotFoundError: if the metric is unknown.
    """
    try:
        metric = metric_funcs[metric_id]
    except KeyError as e:
        raise MetricNotFoundError(f"Metric {metric_id} not found") from e
    return metric


register("mean_directional_accuracy", metrics.mean_directional_accuracy)
register("prediction_trend_accuracy", metrics.prediction_trend_accuracy)
register("r_square", metrics.r_square)
register("mean_absolute_error", metrics.mean_absolute_error)
register("mean_squared_error", metrics.mean_squared_error)
register("mean_absolute_percentage_error", metrics.mean_absolute_percentage_error)
register("root_mean_squared_error", metrics.root_mean_squared_error)
register("root_mean_squared_log_error", metrics.rmsle)
register("mean_squared_log_error", metrics.msle)
register(
    "normalized_root_mean_squared_error", metrics.normalized_root_mean_squared_error
)
register("mean_absolute_scaled_error", metrics.mean_absolute_scaled_error)
