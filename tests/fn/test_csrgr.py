"""Test csrgr."""
import numpy as np
import pytest
from morie.fn.csrgr import csrgr


def test_csrgr_basic():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = csrgr(incidents=inc, population=pop, n=20)
    assert r.value is not None


def test_csrgr_description():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = csrgr(incidents=inc, population=pop, n=20)
    assert r.name
