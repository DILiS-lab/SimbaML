"""Provides the class to create an actual model of a simulation problem."""

import typing
import numpy as np
from scipy import integrate
import pandas as pd

from simba_ml.simulation.sparsifier import sparsifier as abstract_sparsifier
from simba_ml.simulation.sparsifier import no_sparsifier
from simba_ml.simulation import noisers
from simba_ml.simulation import species
from simba_ml.simulation import derivative_noiser
from simba_ml.simulation import distributions
from simba_ml.simulation import kinetic_parameters as kinetic_parameters_module


KineticParameterType = typing.TypeVar("KineticParameterType")


# pylint: disable=too-many-instance-attributes
class SystemModel:
    """A SystemModel is the actual model of a problem.

    This class solves a system of Ordinary Differential Equations(ODEs)
    and generates a signal for a specified number of timestamps.
    The signal can be modified by appliying noise or excluding a
    fraction of data points. The result can be exported as a csv-file.

    Attributes:
        kinetic_parameters: The kinetic parameters of the model.
        sparsifier: A sparsifier randomly removes values from the signal.
            Can be None.
        noiser: A noisers applies noise to the signal. Can be None.
        timestamps: The number of timestamps the model should be capable
            of generating data for.
        solver_method: The solver used to solve ODEs.
            Corresponds to scipy.integrate.solve_ivp(). Default is `LSODA`.
        atol: Absolute tolerance passed to scipy.integrate.solve_ivp().
        rtol: Relative tolerance passed to scipy.integrate.solve_ivp().
    """

    # pylint: disable=too-many-arguments

    @property
    def deriv(
        self,
    ) -> typing.Callable[
        [float, list[float], dict[str, KineticParameterType]], tuple[float, ...]
    ]:
        """The derivative of the model.

        Returns:
            A function that takes a time, a list of species values and
            a dictionary of arguments and returns a tuple of derivatives.
        """
        return self.__deriv

    @property
    def specieses(self) -> dict[str, species.Species]:
        """The species of the model.

        Returns:
            A dictionary of specieses, where the key is the name of the species.
        """
        return self.__specieses

    @property
    def name(self) -> str:
        """The name of the problem.

        Returns:
            The name of the problem.
        """
        return self.__name

    @property
    def kinetic_parameters(
        self,
    ) -> dict[str, kinetic_parameters_module.KineticParameter[KineticParameterType]]:
        """The model arguments.

        Returns:
            A dictionary of arguments, where the key is the name of the argument.
        """
        return self.__kinetic_parameters

    def __init__(
        self,
        name: str,
        specieses: list[species.Species],
        kinetic_parameters: dict[
            str, kinetic_parameters_module.KineticParameter[KineticParameterType]
        ],
        deriv: typing.Callable[
            [float, list[float], dict[str, KineticParameterType]], tuple[float, ...]
        ],
        sparsifier: abstract_sparsifier.Sparsifier | None = None,
        noiser: noisers.Noiser | None = None,
        deriv_noiser: derivative_noiser.DerivNoiser[KineticParameterType] | None = None,
        timestamps: distributions.Distribution[float] | None = None,
        solver_method: str = "LSODA",
        atol: float = 1e-6,
        rtol: float = 1e-3,
    ):
        """Inits PredictionTask with the provided params.

        Args:
            name: The name of the problem.
            specieses: A list of specieses.
            kinetic_parameters: A dictionary of arguments,
                where the key is the name of the argument.
            deriv: A function that takes a time, a list of species values
                and a dictionary of arguments and returns a tuple of derivatives.
            sparsifier: A sparsifier randomly removes values from the signal.
                Can be None.
            noiser: A noisers applies noise to the signal. Can be None.
            deriv_noiser: A special noisers which applies noise to the derivative.
                Can be None.
            timestamps: The number of timestamps the model should be capable
                of generating data for .
            solver_method: The solver used to solve ODEs.
                Corresponds to scipy.integrate.solve_ivp(). Default is `LSODA`.
            atol: Absolute tolerance passed to scipy.integrate.solve_ivp().
                Default is 1e-6.
            rtol: Relative tolerance passed to scipy.integrate.solve_ivp().
                Default is 1e-3.
        """
        self.__name = name
        self.__specieses: dict[str, species.Species] = {
            species.name: species for species in specieses
        }
        self.__kinetic_parameters = kinetic_parameters
        self.sparsifier = sparsifier or no_sparsifier.NoSparsifier()
        self.noiser = noiser or noisers.NoNoiser()
        self.deriv_noiser = deriv_noiser or derivative_noiser.NoDerivNoiser()
        self.timestamps = timestamps or distributions.Constant(1500.0)
        self.__deriv = deriv
        self.solver_method = solver_method
        self.atol = atol
        self.rtol = rtol

    def __get_t(self, timestamps: int | None = None) -> np.typing.NDArray[np.float64]:
        """Returns an array of timestamps in range `[0, timestamps)`.

        Args:
            timestamps(int): The number of timestamps.
                If None, the number of timestamps is sampled ramdomly.

        Returns:
            np.ndarray: Array with timestamps.
        """
        timestamps = timestamps or int(
            distributions.get_random_value_from_distribution(self.timestamps)
        )
        return np.linspace(0, timestamps - 1, timestamps)

    def __hide(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Hides species according to problem file.

        Args:
            signal(pd.DataFrame): Input signal.

        Returns:
            np.DataFrame: DataFrame containing species according
                to problem output definition.
        """
        contained_specieses = [
            name
            for name, species in self.specieses.items()
            if species.contained_in_output
        ]
        modified_df = signal[contained_specieses]
        modified_df.sort_index(inplace=True)
        return modified_df

    def apply_noisifier(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Applies the objects noisifier to a signal.

        Args:
            signal: (pd.DataFrame) The signal.

        Returns:
            pd.DataFrame: Signal with applied noise.
        """
        signal = self.noiser.noisify(signal)
        return signal

    def sample_start_values_from_hypercube(self, n: int) -> dict[str, typing.Any]:
        """Samples the start values of the problem.

        Args:
            n: number of start_values to create

        Start values are:
        - number of individuals per species
        - kinetic_parameters
        - number of timestamps

        Returns:
            dict[str, typing.Any]: The start values dict.
        """
        timestamps = list(map(int, self.timestamps.get_samples_from_hypercube(n)))
        for kinetic_parameter in self.kinetic_parameters.values():
            kinetic_parameter.prepare_samples(n)
        return {
            "specieses": self.sample_species_start_values_from_hypercube(n),
            "timestamps": timestamps,
        }

    def sample_species_start_values_from_hypercube(
        self, n: int
    ) -> dict[str, list[float]]:
        """Samples the start values for the specieses.

        Args:
            n: the number of samples.

        Returns:
            The start values for the specieses, sampled from a hypercube.
        """
        return {
            name: s.get_initial_values_from_hypercube_sampling(n)
            for name, s in self.specieses.items()
        }

    def apply_sparsifier(self, signal: pd.DataFrame) -> pd.DataFrame:
        """Applies the objects sparsifier to a signal.

        Args:
            signal: (pd.DataFrame) The signal.

        Returns:
            pd.DataFrame: Signal of reduced features.
        """
        signal = self.__hide(signal)
        signal = self.sparsifier.sparsify(signal)
        return signal

    def __get_deriv_function(
        self, max_t: int, noised: bool = True
    ) -> typing.Callable[
        [float, list[float], dict[str, KineticParameterType]], tuple[float, ...]
    ]:
        """Returns the derivative function.

        Args:
            max_t: The maximum time the derivative function should be able to handle.
            noised: Whether the derivative function should be noised.

        Returns:
            The derivative function.
        """
        return (
            self.deriv_noiser.noisify(self.__deriv, max_t) if noised else self.__deriv
        )

    def __decorate_deriv(
        self,
        run_number: int,
        deriv: typing.Callable[
            [float, list[float], dict[str, KineticParameterType]],
            tuple[typing.Any, ...],
        ],
    ) -> typing.Callable[
        [
            float,
            list[float],
            dict[str, kinetic_parameters_module.KineticParameter[KineticParameterType]],
        ],
        tuple[float, ...],
    ]:
        """Decorates the (noised) deriv-function, which is provided by the user.

        The deriv-function provided by the user takes the kinetic parameters
        as a dictionary mapping from the name (`str`) to the value (`float`) at this
        timestep. This decorator accesses the `KineticParameter` objects, and asks for
        the values at the timestep t.

        Args:
            run_number: The number of the simulation run
                (used to get the correct kinetic_parameter values).
            deriv: The derivative function, which takes the kinetic parameters
                as a dict mapping their name to a concrete value.

        Returns:
            The decorated derivative function, which takes the kinetic parameters
            as a dict mapping their name to a KineticParameter object.
        """

        def new_deriv(
            t: float,
            y: list[float],
            kinetic_parameters: dict[
                str, kinetic_parameters_module.KineticParameter[KineticParameterType]
            ],
        ) -> tuple[float, ...]:
            return deriv(
                t,
                y,
                {
                    name: kinetic_parameter.get_at_timestamp(run_number, t)
                    for name, kinetic_parameter in kinetic_parameters.items()
                },
            )

        return new_deriv

    def get_clean_signal(
        self,
        start_values: dict[str, typing.Any],
        sample_id: int,
        deriv_noised: bool = True,
    ) -> pd.DataFrame:
        """Creates a clean signal.

        Args:
            start_values: Start values for the simulation.
            sample_id: The id of the sample.
            deriv_noised: Whether the derivative function should be noised.

        Returns:
            A clean signal (possibly the deriv function is noised)
        """
        t = self.__get_t(start_values["timestamps"][sample_id])
        y0 = [
            start_values["specieses"][species_name][sample_id]
            for species_name in self.specieses.keys()
        ]
        deriv_function = self.__get_deriv_function(max_t=max(t), noised=deriv_noised)
        decorated_deriv_function = self.__decorate_deriv(sample_id, deriv_function)

        ret = integrate.solve_ivp(
            fun=decorated_deriv_function,
            y0=y0,
            t_span=(t[0], t[-1]),
            t_eval=t,
            method=self.solver_method,
            args=(self.kinetic_parameters,),
            atol=self.atol,
            rtol=self.rtol,
        )

        return (
            pd.DataFrame(ret.y)
            .set_axis(list(self.specieses.keys()), axis=0)
            .transpose()
        )
