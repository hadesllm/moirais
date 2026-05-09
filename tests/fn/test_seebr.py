"""Test seebr."""
import numpy as np
import pytest
from moirais.fn.seebr import seebr


def test_seebr_basic():
    rng = np.random.default_rng(42)
    cases = rng.poisson(5, 20)
    pop = rng.poisson(1000, 20) + 100
    coords = rng.uniform(0, 100, (20, 2))
    r = seebr(cases=cases, population=pop, coords=coords, n=20)
    assert r.value is not None


def test_seebr_description():
    rng = np.random.default_rng(42)
    cases = rng.poisson(5, 20)
    pop = rng.poisson(1000, 20) + 100
    coords = rng.uniform(0, 100, (20, 2))
    r = seebr(cases=cases, population=pop, coords=coords, n=20)
    assert r.name
