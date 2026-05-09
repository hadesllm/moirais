"""Test cscld."""
import numpy as np
import pytest
from moirais.fn.cscld import cscld


def test_cscld_basic():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = cscld(incidents=inc, population=pop, n=20)
    assert r.value is not None


def test_cscld_description():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = cscld(incidents=inc, population=pop, n=20)
    assert r.name
