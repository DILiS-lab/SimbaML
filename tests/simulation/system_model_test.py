import os

import pandas as pd

from simba_ml.simulation import noisers
from simba_ml.simulation import distributions
from simba_ml.simulation.sparsifier.random_sample_sparsifier import (
    RandomSampleSparsifier,
)
from simba_ml.simulation import system_model
from simba_ml.simulation.generators import time_series_generator
from simba_ml.simulation.species import Species
from simba_ml.simulation import kinetic_parameters as kinetic_parameters_module


def delete_dir_if_empty(directory):
    if len(os.listdir(directory)) == 0:
        os.rmdir(directory)


def test_get_clean_signal_has_correct_length():
    timestamps = 200
    sm = create_example_prediciton_task(timestamps)
    start_values = sm.sample_start_values_from_hypercube(1)
    clean_signal = sm.get_clean_signal(start_values, 0)
    assert clean_signal.shape[0] == timestamps


def test_get_clean_signal_correct_data():
    sm = create_example_prediciton_task()
    start_values = sm.sample_start_values_from_hypercube(1)
    clean_signal = sm.get_clean_signal(start_values, 0)
    assert clean_signal["y"].unique().size == 1


def test_apply_noisifier():
    sm = create_example_prediciton_task()
    before = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 3.0, 4.0]})
    expected = pd.DataFrame({"A": [2.0, 3.0, 4.0], "B": [3.0, 4.0, 5.0]})
    noised_signal = sm.apply_noisifier(before)
    assert noised_signal.equals(expected)


def test_apply_sparsifier():
    sm = create_example_prediciton_task()
    sm.sparsifier = RandomSampleSparsifier()
    before = pd.DataFrame(
        {
            "y": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    after = sm.apply_sparsifier(before)

    assert after.shape[0] == int(before.shape[0] / 2)


def test_generate_signal():
    sm = create_example_prediciton_task(5)
    generator = time_series_generator.TimeSeriesGenerator(sm)

    expected = pd.DataFrame({"y": [1.0, 1.0, 1.0, 1.0, 1.0]}).reset_index(drop=True)

    after = generator.generate_signal().reset_index(drop=True)
    assert after.equals(expected)


def test_generate_csv():
    sm = create_example_prediciton_task(5)
    generator = time_series_generator.TimeSeriesGenerator(sm)

    expected = pd.DataFrame({"y": [1.0, 1.0, 1.0, 1.0, 1.0]}).reset_index(drop=True)

    generator.generate_csv()
    actual = pd.read_csv("./data/Constant function_0.csv").reset_index(drop=True)
    os.remove("./data/Constant function_0.csv")
    delete_dir_if_empty("./data/")

    assert expected.equals(actual)


def test_generate_csvs():
    sm = create_example_prediciton_task(5)
    generator = time_series_generator.TimeSeriesGenerator(sm)

    expected = pd.DataFrame({"y": [1.0, 1.0, 1.0, 1.0, 1.0]}).reset_index(drop=True)

    generator.generate_csvs(n=1)
    actual = pd.read_csv("./data/Constant function_0.csv").reset_index(drop=True)
    os.remove("./data/Constant function_0.csv")
    delete_dir_if_empty("./data/")

    assert expected.equals(actual)


def test_get_signals_returns_dataframes():
    sm = create_example_prediciton_task()
    generator = time_series_generator.TimeSeriesGenerator(sm)
    signals = generator.generate_signals(n=3)
    assert isinstance(signals, list)
    assert all(isinstance(signal, pd.DataFrame) for signal in signals)


def test_has_kinetic_parameters():
    sm = create_example_prediciton_task()
    assert len(sm.kinetic_parameters) == 1


def create_example_prediciton_task(number_of_timestamps=200):
    name = "Constant function"
    timestamps = distributions.Constant(number_of_timestamps)
    specieses = [Species("y", distributions.Constant(0))]
    kinetic_parameters = {
        "Useless": kinetic_parameters_module.ConstantKineticParameter(
            distributions.Constant(1)
        )
    }
    solver_method = "RK45"
    atol = 1e-8
    rtol = 1e-5

    def deriv(_t, _y, _arguments):
        """Derivative of the function at the point _t.

        Returns:
            List[float]
        """
        return [0]

    noiser = noisers.additive_noiser.AdditiveNoiser(
        distributions.NormalDistribution(1, 0)
    )
    sm = system_model.SystemModel(
        name,
        specieses,
        kinetic_parameters,
        deriv=deriv,
        noiser=noiser,
        timestamps=timestamps,
        solver_method=solver_method,
        atol=atol,
        rtol=rtol,
    )
    return sm
