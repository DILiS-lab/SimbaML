"""Provides a random generator."""
import numpy as np

rng = [np.random.default_rng(0)]


def get_rng() -> np.random.Generator:
    """Returns the random generator.

    Returns:
      The random generator
    """
    return rng[0]


def set_seed(seed: int) -> None:
    """Sets the seed.

    Attributes:
      seed: the randoms seed
    """
    rng[0] = np.random.default_rng(seed)
