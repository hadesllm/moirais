"""Test sepvl."""
import numpy as np
import pytest
from morie.fn.sepvl import sepvl


def test_sepvl_basic():
    rng = np.random.default_rng(42)
    cases = rng.poisson(5, 20)
    pop = rng.poisson(1000, 20) + 100
    coords = rng.uniform(0, 100, (20, 2))
    r = sepvl(cases=cases, population=pop, coords=coords, n=20)
    assert r.value is not None


def test_sepvl_description():
    rng = np.random.default_rng(42)
    cases = rng.poisson(5, 20)
    pop = rng.poisson(1000, 20) + 100
    coords = rng.uniform(0, 100, (20, 2))
    r = sepvl(cases=cases, population=pop, coords=coords, n=20)
    assert r.name
