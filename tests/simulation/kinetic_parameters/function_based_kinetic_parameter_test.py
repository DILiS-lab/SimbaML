import itertools
import pytest

from simba_ml.simulation import kinetic_parameters


@pytest.mark.parametrize("run,t", itertools.product(range(10), range(1, 5)))
def test_function_based_kinetic_parameter(run, t):
    def function(_t):
        return 1

    kinetic_parameter = kinetic_parameters.FunctionBasedKineticParameter(function)
    kinetic_parameter.prepare_samples(10)
    assert kinetic_parameter.get_at_timestamp(run, t) == 1


@pytest.mark.parametrize("run,t", itertools.product(range(10), range(1, 5)))
def test_function_based_kinetic_parameter_with_uniform_distribution(run, t):
    def function(_t):
        return t * 2

    kinetic_parameter = kinetic_parameters.FunctionBasedKineticParameter(function)
    kinetic_parameter.prepare_samples(10)
    assert kinetic_parameter.get_at_timestamp(run, t) == t * 2
