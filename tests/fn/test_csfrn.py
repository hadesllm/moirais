"""Test csfrn."""
import numpy as np
import pytest
from morie.fn.csfrn import csfrn


def test_csfrn_basic():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = csfrn(incidents=inc, population=pop, n=20)
    assert r.value is not None


def test_csfrn_description():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = csfrn(incidents=inc, population=pop, n=20)
    assert r.name
