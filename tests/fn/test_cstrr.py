"""Test cstrr."""
import numpy as np
import pytest
from moirais.fn.cstrr import cstrr


def test_cstrr_basic():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = cstrr(incidents=inc, population=pop, n=20)
    assert r.value is not None


def test_cstrr_description():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = cstrr(incidents=inc, population=pop, n=20)
    assert r.name
