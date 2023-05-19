import pytest

from simba_ml.simulation import derivative_noiser
from simba_ml.simulation import distributions


@pytest.mark.parametrize("y,t", [([1, 2, 3], 2), ([1, 2, 3], 3), ([1, 2, 3, 4, 5], 3)])
def test_no_deriv_noiser(y, t):
    """Check that the `NoDerivNoiser` does not put any noise to the derivate."""

    def before(t, y, _kinetic_parameters):
        return tuple(v * t for v in y)

    after = derivative_noiser.NoDerivNoiser().noisify(before, 5)
    assert before(t, y, {}) == after(t, y, {})


@pytest.mark.parametrize("y,t", [([1, 2, 3], 2), ([1, 2, 3], 3), ([1, 2, 3, 4, 5], 3)])
def test_additional_deriv_noiser_with_sigma_equals_1(y, t):
    """Check, that the `AdditiveDerivNoiser` puts Noise to the derivate."""

    def before(t, y, _kinetic_parameters):
        return tuple(v * t for v in y)

    after = derivative_noiser.AdditiveDerivNoiser(
        distributions.NormalDistribution(0, 1)
    ).noisify(before, 5)
    assert before(t, y, {}) != after(t, y, {})


@pytest.mark.parametrize("y,t", [([1, 2, 3], 2), ([1, 2, 3], 3), ([1, 2, 3, 4, 5], 3)])
def test_additional_deriv_noiser_with_sigma_equals_0(y, t):
    """Check, that the `AdditiveDerivNoiser` puts Noise to the derivate."""

    def before(t, y, _kinetic_parameters):
        return tuple(v * t for v in y)

    after = derivative_noiser.AdditiveDerivNoiser(
        distributions.NormalDistribution(0, 0)
    ).noisify(before, 5)
    assert before(t, y, {}) == after(t, y, {})


@pytest.mark.parametrize("y,t", [([1, 2, 3], 2), ([1, 2, 3], 3), ([1, 2, 3, 4, 5], 3)])
def test_multiplicative_deriv_noiser_with_sigma_equals_1(y, t):
    """Check, that the `MultiplicativeDerivNoiser` puts Noise to the derivate."""

    def before(t, y, _kinetic_parameters):
        return tuple(v * t for v in y)

    after = derivative_noiser.MultiplicativeDerivNoiser(
        distributions.NormalDistribution(1, 1)
    ).noisify(before, 5)
    assert before(t, y, {}) != after(t, y, {})


@pytest.mark.parametrize("y,t", [([1, 2, 3], 2), ([1, 2, 3], 3), ([1, 2, 3, 4, 5], 3)])
def test_multiplicative_deriv_noiser_with_sigma_equals_0(y, t):
    """Check, that the `MultiplicativeDerivNoiser`
    puts Noise to the derivate."""

    def before(t, y, _kinetic_parameters):
        return tuple(v * t for v in y)

    after = derivative_noiser.MultiplicativeDerivNoiser(
        distributions.NormalDistribution(1, 0)
    ).noisify(before, 5)
    assert before(t, y, {}) == after(t, y, {})


@pytest.mark.parametrize("y,t", [([1, 2, 3], 2), ([1, 2, 3], 3), ([1, 2, 3, 4, 5], 5)])
def test_multi_deriv_noiser(y, t):
    """Check, that the `MultiplicativeDerivNoiser` puts Noise to the derivate
    by appliyng one of the given `Noiser`."""

    def before(t, y, _kinetic_parameters):
        return tuple(v * t for v in y)

    noiser1 = derivative_noiser.MultiplicativeDerivNoiser(
        distributions.NormalDistribution(2, 0)
    )
    noiser2 = derivative_noiser.AdditiveDerivNoiser(
        distributions.NormalDistribution(1, 0)
    )
    noiser = derivative_noiser.MultiDerivNoiser([noiser1, noiser2])
    after = noiser.noisify(deriv=before, max_t=5)
    assert after(t, y, {}) in [
        noiser.noisify(before, 5)(t, y, {}) for noiser in noiser.noisers
    ]


@pytest.mark.parametrize("y,t", [([1, 2, 3], 2), ([1, 2, 3], 3), ([1, 2, 3, 4, 5], 5)])
def test_sequential_deriv_noiser(y, t):
    """Check, that the `SequentialDerivNoiser` puts Noise to the derivate
    by appliyng the given `DerivNoiser` sequentially."""

    def before(t, y, _kinetic_parameters):
        return tuple(v * t for v in y)

    noiser1 = derivative_noiser.MultiplicativeDerivNoiser(
        distributions.NormalDistribution(2, 0)
    )
    noiser2 = derivative_noiser.AdditiveDerivNoiser(
        distributions.NormalDistribution(1, 0)
    )
    noiser = derivative_noiser.SequentialDerivNoiser([noiser1, noiser2])
    after = noiser.noisify(deriv=before, max_t=5)
    assert after(t, y, {}) == noiser2.noisify(noiser1.noisify(before, 5), 5)(t, y, {})
