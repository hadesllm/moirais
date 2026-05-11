"""Test cscpt2."""
import numpy as np
import pytest
from morie.fn.cscpt2 import cscpt2


def test_cscpt2_basic():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = cscpt2(incidents=inc, population=pop, n=20)
    assert r.value is not None


def test_cscpt2_description():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = cscpt2(incidents=inc, population=pop, n=20)
    assert r.name
