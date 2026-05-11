"""Test csdmv."""
import numpy as np
import pytest
from morie.fn.csdmv import csdmv


def test_csdmv_basic():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = csdmv(incidents=inc, population=pop, n=20)
    assert r.value is not None


def test_csdmv_description():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = csdmv(incidents=inc, population=pop, n=20)
    assert r.name
