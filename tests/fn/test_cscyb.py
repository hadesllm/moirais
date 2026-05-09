"""Test cscyb."""
import numpy as np
import pytest
from moirais.fn.cscyb import cscyb


def test_cscyb_basic():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = cscyb(incidents=inc, population=pop, n=20)
    assert r.value is not None


def test_cscyb_description():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = cscyb(incidents=inc, population=pop, n=20)
    assert r.name
