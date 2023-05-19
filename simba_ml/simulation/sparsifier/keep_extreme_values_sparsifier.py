"""Removes a given relative amount of samples from a signal."""

import pandas as pd

from simba_ml import error_handler
from simba_ml.simulation.sparsifier import sparsifier as sparsifier_module


class KeepExtremeValuesSparsifier(sparsifier_module.Sparsifier):
    """A `Sparsifier` that keeps extreme values."""

    def __init__(
        self,
        sparsifier: sparsifier_module.Sparsifier,
        lower_bound: float = 0.1,
        upper_bound: float = 0.1,
    ):
        """Inits the `KeepExtremeValuesSparsifier`.

        Args:
            sparsifier: The sparsifier to apply to the signal.
            lower_bound: The fraction of timestamps to keep because the values
                is in the lower bound.
            upper_bound: The fraction of timestamps to keep because the values
                is in the upper bound.


        Raises:
            ValueError: lower_bound or upper_bound is not in range [0, 1]
                or lower_bound > upper_bound.
            TypeError: lower_bound or upper_bound is not a float.

        Examples:
            >>> import pandas as pd
            >>> from simba_ml.simulation import sparsifier
            >>> signal = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
            >>> sparsifier.keep_extreme_values_sparsifier.KeepExtremeValuesSparsifier(
            ...     sparsifier.random_sample_sparsifier.RandomSampleSparsifier(0)
            ... ).sparsify(signal).sort_index()
                a
            0   1
            9  10
            >>> signal = pd.DataFrame({
            ...     "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            ...     "b": [1, 3, 5, 7, 9, 10, 8, 6, 5, 2]})
            >>> sparsifier.keep_extreme_values_sparsifier.KeepExtremeValuesSparsifier(
            ...     sparsifier.random_sample_sparsifier.RandomSampleSparsifier(0),
            ... ).sparsify(signal).sort_index()
                a   b
            0   1   1
            5   6  10
            9  10   2
        """
        error_handler.confirm_param_is_float(
            param=lower_bound, param_name="lower_bound"
        )
        error_handler.confirm_param_is_float(
            param=upper_bound, param_name="upper_bound"
        )
        error_handler.confirm_number_is_in_interval(
            lower_bound, param_name="lower_bound", start_value=0.0, end_value=1.0
        )
        error_handler.confirm_number_is_in_interval(
            upper_bound, param_name="upper_bound", start_value=lower_bound, end_value=1
        )

        self.sparsifier = sparsifier
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __create_extreme_value_column(self, signal: pd.DataFrame) -> str:
        name_of_extreme_value_column = "extreme_value"
        while name_of_extreme_value_column in signal.columns:
            name_of_extreme_value_column += "_"
        signal[name_of_extreme_value_column] = False
        return name_of_extreme_value_column

    def sparsify(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Removes some (`1-frac`) samples chosen with a uniform random distributions.

        Args:
            signal: The signal to sparsify.

        Returns:
            DataFrame: The sparsified signal.
        """
        name_of_extreme_value_column = self.__create_extreme_value_column(signal)
        for column in signal.columns:
            if column == name_of_extreme_value_column:
                continue
            signal = signal.sort_values(column)
            assert name_of_extreme_value_column == signal.columns[-1]
            signal.iloc[: int(self.lower_bound * len(signal)), -1] = True
            signal.iloc[int((1 - self.upper_bound) * len(signal)) :, -1] = True

        sparsed_signal = pd.concat(
            [
                signal[signal[name_of_extreme_value_column]],
                self.sparsifier.sparsify(signal[signal[name_of_extreme_value_column]]),
            ]
        )
        return sparsed_signal.drop(columns=[name_of_extreme_value_column]).sort_index()
