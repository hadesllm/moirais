"""Test cscpt."""
import numpy as np
import pytest
from morie.fn.cscpt import cscpt


def test_cscpt_basic():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = cscpt(incidents=inc, population=pop, n=20)
    assert r.value is not None


def test_cscpt_description():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = cscpt(incidents=inc, population=pop, n=20)
    assert r.name
