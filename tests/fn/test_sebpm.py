"""Test sebpm."""
import numpy as np
import pytest
from moirais.fn.sebpm import sebpm


def test_sebpm_basic():
    rng = np.random.default_rng(42)
    cases = rng.poisson(5, 20)
    pop = rng.poisson(1000, 20) + 100
    coords = rng.uniform(0, 100, (20, 2))
    r = sebpm(cases=cases, population=pop, coords=coords, n=20)
    assert r.value is not None


def test_sebpm_description():
    rng = np.random.default_rng(42)
    cases = rng.poisson(5, 20)
    pop = rng.poisson(1000, 20) + 100
    coords = rng.uniform(0, 100, (20, 2))
    r = sebpm(cases=cases, population=pop, coords=coords, n=20)
    assert r.name
