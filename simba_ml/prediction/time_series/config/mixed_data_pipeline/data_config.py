"""Provides the configuration for the data."""
import typing
import dataclasses
from typing import Optional
from simba_ml.prediction.time_series.config import (
    time_series_config,
)


@dataclasses.dataclass
class DataConfig:
    """Config for time-series data."""

    ratios: list[float]
    time_series: time_series_config.TimeSeriesConfig
    observed: Optional[str] = None
    synthetic: Optional[str] = None
    test_split: float = 0.2
    split_axis: str = "vertical"
    input_features: typing.Optional[list[str]] = None
    output_features: typing.Optional[list[str]] = None
    export_path: typing.Optional[str] = None
