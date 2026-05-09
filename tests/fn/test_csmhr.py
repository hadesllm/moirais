"""Test csmhr."""
import numpy as np
import pytest
from moirais.fn.csmhr import csmhr


def test_csmhr_basic():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = csmhr(incidents=inc, population=pop, n=20)
    assert r.value is not None


def test_csmhr_description():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = csmhr(incidents=inc, population=pop, n=20)
    assert r.name
