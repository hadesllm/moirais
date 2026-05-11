"""Test cscir."""
import numpy as np
import pytest
from morie.fn.cscir import cscir


def test_cscir_basic():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = cscir(incidents=inc, population=pop, n=20)
    assert r.value is not None


def test_cscir_description():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = cscir(incidents=inc, population=pop, n=20)
    assert r.name
