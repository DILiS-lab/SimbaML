import math
import numpy as np
import pytest

from simba_ml.simulation import distributions


def test_distribution_get_random_value():
    assert (
        distributions.get_random_value_from_distribution(distributions.Constant(1)) == 1
    )


@pytest.mark.parametrize("mu, sigma, n", [(0, 1, 50), (-100, 10, 100), (10, 10, 1000)])
def test_normal_distribution_get_samples_from_hypercube(mu, sigma, n):
    """Check that the `get_samples_from_hypercube` function returns a list of floats
    with sensible content for normal distributions."""
    samples = distributions.NormalDistribution(mu, sigma).get_samples_from_hypercube(n)
    assert math.isclose(np.mean(samples), mu, abs_tol=sigma * 0.2)
    assert math.isclose(np.var(samples), sigma**2, rel_tol=0.2)
    assert len(samples) == n


def test_beta_distribution_get_samples_from_hypercube():
    """Check that the `get_samples_from_hypercube` function returns a list of floats
    with sensible content for normal distribution."""
    n = 1000
    alpha = 1
    beta = 1
    samples = distributions.BetaDistribution(alpha, beta).get_samples_from_hypercube(n)
    assert len(samples) == n
    mean = alpha / (alpha + beta)
    variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    assert math.isclose(np.mean(samples), mean, abs_tol=0.1)
    assert math.isclose(np.var(samples), variance, rel_tol=0.2)


def test_lognormal_distribution_get_samples_from_hypercube():
    """Check that the `get_samples_from_hypercube` function returns a list of floats
    with sensible content for normal distribution."""
    n = 100000
    mu = 0
    sigma = 1
    samples = distributions.LogNormalDistribution(mu, sigma).get_samples_from_hypercube(
        n
    )
    assert len(samples) == n
    mean = math.e ** (mu + sigma / 2)
    variance = (math.e**sigma - 1) * math.e ** (2 * mu + sigma)
    assert math.isclose(np.mean(samples), mean, abs_tol=0.1)
    assert math.isclose(np.var(samples), variance, rel_tol=0.2)


def test_scalar_distribution_one_value():
    """Check that the `Constant` returns the given scalar when calling
    `Constant.get_random_value`."""
    value = 1.0
    distribution = distributions.Constant(value)
    assert value == distributions.get_random_value_from_distribution(distribution)


def test_scalar_distribution_multiple_values():
    """Check that the `Constant` returns the given scalar in the provided shape when
    calling `Constant.get_multiple_random_values`."""
    value = 4.2
    distribution = distributions.Constant(value)
    expected = [4.2, 4.2, 4.2]
    result = distribution.get_random_values(3)
    assert np.array_equal(expected, result)


def test_scalar_distribution_get_samples_from_hypercube():
    value = 4.2
    distribution = distributions.Constant(value)
    expected = [4.2, 4.2, 4.2]
    result = distribution.get_samples_from_hypercube(3)
    assert np.array_equal(expected, result)


def test_vector_distribution_one_value():
    """Check that the `VectorDistribution` returns a scalar contained by the given
    vector when calling `VectorDistribution.get_random_value`."""
    values = [1.0, 20.0, 100.0]
    initial_condition = distributions.VectorDistribution(values)
    assert distributions.get_random_value_from_distribution(initial_condition) in values


def test_vector_distribution_multiple_values():
    """Check that the `VectorDistribution` returns a valid output when calling
    `get_multiple_random_values`"""
    values = [1.0, 20.0, 100.0]
    initial_condition = distributions.VectorDistribution(values)
    n = 5
    result = initial_condition.get_random_values(5)
    assert all(n in values for n in result) and len(result) == n


def test_vector_distribution_get_samples_from_hypercube():
    values = [1.0, 20.0, 100.0]
    initial_condition = distributions.VectorDistribution(values)
    result = initial_condition.get_samples_from_hypercube(30)
    assert len(result) == 30
    assert 10 <= result.count(values[0]) <= 11
    assert 10 <= result.count(values[1]) <= 11
    assert 10 <= result.count(values[2]) <= 11


def test_vector_distribution_throws_index_error_when_values_array_is_empty():
    """Check that the `VectorDistribution` throws an IndexError when initialized
    with an empy array."""
    with pytest.raises(IndexError):
        distributions.VectorDistribution(values=[])


def test_vector_distribution_throws_error_if_values_contains_element_no_float_or_int():
    """Check that the `VectorDistribution` throws an TypeError when initialized
    with values not float or int."""
    with pytest.raises(TypeError):
        distributions.VectorDistribution(values=[3, 2, "7"])


def test_continuous_uniform_distribution_one_value():
    """Check that the `ContinuousUniformDistribution` returns a scalar inside
    of the given range when calling `ContinuousUniformDistribution.get_random_value`."""
    my_min = 3
    my_max = 5

    initial_condition = distributions.ContinuousUniformDistribution(my_min, my_max)
    assert (
        my_min
        <= distributions.get_random_value_from_distribution(initial_condition)
        <= my_max
    )


def test_continuous_uniform_distribution_multiple_values():
    """Check that the `ContinuousUniformDistribution` returns an array containing
    values from the given range when calling `get_multiple_random_values`."""
    my_min = 3
    my_max = 5

    n = 5
    initial_condition = distributions.ContinuousUniformDistribution(my_min, my_max)
    result = initial_condition.get_random_values(n)
    assert all(my_min <= n <= my_max for n in result) and len(result) == n


def test_continuous_uniform_distribution_throws_error_when_min_value_not_float_or_int():
    """Check that the `ContinuousUniformDistribution` throws a TypeError when min_value
    is not float or int."""
    with pytest.raises(TypeError):
        distributions.ContinuousUniformDistribution("1", -1)


def test_continuous_uniform_distribution_throws_error_when_max_value_not_float_or_int():
    """Check that the `ContinuousUniformDistribution` throws a TypeError when max_value
    is not float or int."""
    with pytest.raises(TypeError):
        distributions.ContinuousUniformDistribution(1, "3")


def test_continuous_get_hypercube_samples():
    initial_condition = distributions.ContinuousUniformDistribution(3, 6)
    result = sorted(initial_condition.get_samples_from_hypercube(3))
    assert len(result) == 3
    assert 3 <= result[0] <= 4
    assert 4 <= result[1] <= 5
    assert 5 <= result[2] <= 6


def test_beta_distribution_one_value():
    """Check that the `BetaDistribution` returns values from a beta distributions
    when calling `get_random_value`."""
    alpha, beta = 4, 2
    initial_condition = distributions.BetaDistribution(alpha, beta)
    return_values = [
        distributions.get_random_value_from_distribution(initial_condition)
        for _ in range(10000)
    ]
    mean = np.mean(return_values)
    var = np.var(return_values)

    expected_mean = alpha / (alpha + beta)
    expected_var = (alpha * beta) / ((alpha + beta + 1) * (alpha + beta) ** 2)

    assert math.isclose(mean, expected_mean, abs_tol=0.1) and math.isclose(
        var, expected_var, abs_tol=1
    )


def test_beta_distribution_multiple_values():
    """Check that the `BetaDistribution` returns an array containing values
    from a beta distributions when calling `get_multiple_random_values`."""
    alpha, beta = 11, 3
    n = 10000
    initial_condition = distributions.BetaDistribution(alpha, beta)
    return_values = initial_condition.get_random_values(n)
    mean = np.mean(return_values)
    variance = np.var(return_values)

    expected_mean = alpha / (alpha + beta)
    expected_var = (alpha * beta) / ((alpha + beta + 1) * (alpha + beta) ** 2)

    assert math.isclose(expected_mean, mean, abs_tol=0.1)
    assert math.isclose(expected_var, variance, abs_tol=0.1)
    assert len(return_values) == n


def test_beta_distribution_throws_value_error_at_non_positive_alpha():
    """Check that the `BetaDistribution` throws a ValueError
    at non positive values for alpha."""
    with pytest.raises(ValueError):
        distributions.BetaDistribution(-1, 1)


def test_beta_distribution_throws_value_error_at_non_positive_beta():
    """Check that the `BetaDistribution` throws a ValueError
    at non positive values for beta."""
    with pytest.raises(ValueError):
        distributions.BetaDistribution(1, -1)


def test_beta_distribution_throws_type_error_when_alpha_not_float_or_int():
    """Check that the `BetaDistribution` throws a TypeError when alpha is
    not float or int."""
    with pytest.raises(TypeError):
        distributions.BetaDistribution("1", -1)


def test_beta_distribution_throws_type_error_when_beta_not_float_or_int():
    """Check that the `BetaDistribution` throws a TypeError when alpha is
    not float or int."""
    with pytest.raises(TypeError):
        distributions.BetaDistribution(1, "2")


def test_standard_normal_distribution_returns_appr_0_mean_1_var():
    """Check that the `NormalDistribution` with mean 0 and variance 1
    (standard normal distributions) returns values with an approximate mean of 0
    and a variance of 1 when calling `NormalDistribution.get_random_value`."""
    initial_condition = distributions.NormalDistribution(0, 1)
    return_values = [
        distributions.get_random_value_from_distribution(initial_condition)
        for _ in range(10000)
    ]
    mean = np.mean(return_values)
    var = np.var(return_values)
    assert math.isclose(mean, 0, abs_tol=0.1) and math.isclose(var, 1, abs_tol=0.1)


def test_standard_normal_distribution_returns_appr_0_mean_1_var_multiple_values():
    """Check that the `NormalDistribution` with mean 0 and variance 1
    (standard normal distributions) returns values with an approximate mean of 0
    and a variance of 1 when calling`NormalDistribution.get__multiple_random_values`.
    """
    expected_means = [0, 0, 0, 0]
    expected_vars = [1, 1, 1, 1]
    initial_condition = distributions.NormalDistribution(0, 1)
    shape = (4, 10000)
    return_values = initial_condition.get_random_values(shape)

    means = np.mean(return_values, axis=1)
    _vars = np.var(return_values, axis=1)

    assert np.allclose(expected_means, means, atol=0.1) and np.allclose(
        expected_vars, _vars, atol=0.1
    )


def test_normal_distribution_value_error_at_negative_sigma_value():
    """Check that the `NormalDistribution` throws a ValueError
    at negative values for sigma."""
    with pytest.raises(ValueError):
        distributions.NormalDistribution(0, -1)


def test_normal_distribution_throws_type_error_when_mu_is_not_int_or_float():
    """Check that the `NormalDistribution` throws a TypeError
    when mu is not of type int or float."""
    with pytest.raises(TypeError):
        distributions.NormalDistribution("1", 3)


def test_normal_distribution_throws_type_error_when_sigma_is_not_int_or_float():
    """Check that the `NormalDistribution` throws a TypeError
    when sigma is not of type int or float."""
    with pytest.raises(TypeError):
        distributions.NormalDistribution(1, "3")


def test_log_normal_distribution_returns_value_above_0_one_value():
    """Check that the `LogNormalDistribution` returns a scalar inside
    of the given range when calling `LogNormalDistribution.get_random_value`."""
    initial_condition = distributions.LogNormalDistribution(1, 2)
    min_return = min(
        distributions.get_random_value_from_distribution(initial_condition)
        for _ in range(100)
    )
    assert min_return > 0


def test_log_normal_distribution_returns_value_above_0_multiple_values():
    """Check that the `LogNormalDistribution` returns an array of values
    in the given range when calling `get_multiple_random_values`."""
    shape = (4, 2)
    initial_condition = distributions.LogNormalDistribution(1, 2)
    min_returns = np.min(initial_condition.get_random_values(shape), axis=1)

    assert all(v > 0 for v in min_returns.ravel())


def test_log_normal_distribution_throws_value_error_at_negative_sigma_value():
    """Check that the `LogNormalDistribution` throws an ValueError
    at non negative values for sigma."""
    with pytest.raises(ValueError):
        distributions.LogNormalDistribution(0, -1)


def test_log_normal_distribution_throws_type_error_when_mu_is_not_float_or_int():
    """Check that the `LogNormalDistribution` throws a TypeError
    when mu is not of type int or float."""
    with pytest.raises(TypeError):
        distributions.LogNormalDistribution("0", -1)


def test_log_normal_distribution_throws_type_error_when_sigma_is_not_float_or_int():
    """Check that the `LogNormalDistribution` throws a TypeError
    when mu is not of type int or float."""
    with pytest.raises(TypeError):
        distributions.LogNormalDistribution(0, "2")
