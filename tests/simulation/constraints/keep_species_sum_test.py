import math

from simba_ml.simulation import noisers
from simba_ml.simulation import system_model
from simba_ml.simulation.generators import time_series_generator
from simba_ml.simulation.species import Species
from simba_ml.simulation import constraints
from simba_ml.simulation import distributions


def test_constraint_holds_sum():
    timestamps = 200
    pt = create_example_system_model(timestamps)
    generator = time_series_generator.TimeSeriesGenerator(pt)
    signal = generator.generate_signal()
    assert_sums_equal(signal, 1000)


def assert_sums_equal(signal, expected_sum):
    sums = signal.sum(axis=1)
    for s in sums:
        assert math.isclose(s, expected_sum)


def create_example_system_model(number_of_timestamps=200):
    name = "Constant function"
    timestamps = distributions.Constant(number_of_timestamps)

    specieses = [
        Species("y1", distributions.Constant(500)),
        Species("y2", distributions.Constant(500)),
    ]

    kinetic_parameters = {}

    def deriv(_t, _y, _arguments):
        """Derivative of the function at the point _t.

        Returns:
            List[float]
        """
        return [1, 1]

    noiser = noisers.additive_noiser.AdditiveNoiser(
        distributions.NormalDistribution(1, 100)
    )
    pt = constraints.keep_species_sum.KeepSpeciesSum(
        system_model.SystemModel(
            name,
            specieses,
            kinetic_parameters,
            deriv=deriv,
            noiser=noiser,
            timestamps=timestamps,
        ),
        species_sum=1000,
    )
    return pt
