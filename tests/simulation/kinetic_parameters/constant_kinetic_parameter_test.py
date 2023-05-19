import itertools

import pytest

from simba_ml.simulation import kinetic_parameters
from simba_ml.simulation import distributions


@pytest.mark.parametrize("run,t", itertools.product(range(10), range(5)))
def test_constant_kinetic_parameter(run, t):
    """Check that the `ConstantKineticParameter` returns the same value
    for each timestep."""
    kinetic_parameter = kinetic_parameters.ConstantKineticParameter(
        distributions.Constant(1)
    )
    kinetic_parameter.prepare_samples(10)
    assert kinetic_parameter.get_at_timestamp(run, t) == 1


@pytest.mark.parametrize("run, t", itertools.product(range(10), range(1, 5)))
def test_constant_kinetic_parameter_with_uniform_distribution(run, t):
    """Check that the `ConstantKineticParameter` returns the same value
    for each timestep."""
    kinetic_parameter = kinetic_parameters.ConstantKineticParameter(
        distributions.ContinuousUniformDistribution(1, 3)
    )
    kinetic_parameter.prepare_samples(10)
    value = kinetic_parameter.get_at_timestamp(run, 0)
    assert kinetic_parameter.get_at_timestamp(run, t) == value


def test_raises_error_if_get_value_is_called_before_prepare_samples():
    """Check that the `ConstantKineticParameter` returns the same value
    for each timestep."""
    kinetic_parameter = kinetic_parameters.ConstantKineticParameter(
        distributions.ContinuousUniformDistribution(1, 3)
    )
    with pytest.raises(RuntimeError):
        kinetic_parameter.get_at_timestamp(0, 0)


def test_raises_error_if_get_value_is_called_with_too_large_run():
    kinetic_parameter = kinetic_parameters.ConstantKineticParameter(
        distributions.ContinuousUniformDistribution(1, 3)
    )
    kinetic_parameter.prepare_samples(10)
    with pytest.raises(RuntimeError):
        kinetic_parameter.get_at_timestamp(10, 0)


def test_set_for_run_raises_error_if_not_prepared():
    kinetic_parameter = kinetic_parameters.ConstantKineticParameter(
        distributions.ContinuousUniformDistribution(1, 3)
    )
    with pytest.raises(RuntimeError):
        kinetic_parameter.set_for_run(0, 0)


def test_set_for_run_raises_error_if_run_is_too_large():
    kinetic_parameter = kinetic_parameters.ConstantKineticParameter(
        distributions.ContinuousUniformDistribution(1, 3)
    )
    kinetic_parameter.prepare_samples(10)
    with pytest.raises(RuntimeError):
        kinetic_parameter.set_for_run(10, 0)
