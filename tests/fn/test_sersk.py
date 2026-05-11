"""Test sersk."""
import numpy as np
import pytest
from morie.fn.sersk import sersk


def test_sersk_basic():
    rng = np.random.default_rng(42)
    cases = rng.poisson(5, 20)
    pop = rng.poisson(1000, 20) + 100
    coords = rng.uniform(0, 100, (20, 2))
    r = sersk(cases=cases, population=pop, coords=coords, n=20)
    assert r.value is not None


def test_sersk_description():
    rng = np.random.default_rng(42)
    cases = rng.poisson(5, 20)
    pop = rng.poisson(1000, 20) + 100
    coords = rng.uniform(0, 100, (20, 2))
    r = sersk(cases=cases, population=pop, coords=coords, n=20)
    assert r.name
