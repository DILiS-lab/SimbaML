"""Provides a random generator."""
import os
import numpy as np

environment_seed_key = "SIMBA_ML_SEED"
seed = int(os.environ[environment_seed_key]) if environment_seed_key in os.environ else 0
rng = np.random.default_rng(seed)


def get_rng() -> np.random.Generator:
    """Returns the random generator.

    Returns:
      The random generator
    """
    return rng
