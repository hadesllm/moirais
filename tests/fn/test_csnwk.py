"""Test csnwk."""
import numpy as np
import pytest
from moirais.fn.csnwk import csnwk


def test_csnwk_basic():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = csnwk(incidents=inc, population=pop, n=20)
    assert r.value is not None


def test_csnwk_description():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = csnwk(incidents=inc, population=pop, n=20)
    assert r.name
