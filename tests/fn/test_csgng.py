"""Test csgng."""
import numpy as np
import pytest
from morie.fn.csgng import csgng


def test_csgng_basic():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = csgng(incidents=inc, population=pop, n=20)
    assert r.value is not None


def test_csgng_description():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = csgng(incidents=inc, population=pop, n=20)
    assert r.name
