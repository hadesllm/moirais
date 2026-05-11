"""Test csdrg."""
import numpy as np
import pytest
from morie.fn.csdrg import csdrg


def test_csdrg_basic():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = csdrg(incidents=inc, population=pop, n=20)
    assert r.value is not None


def test_csdrg_description():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = csdrg(incidents=inc, population=pop, n=20)
    assert r.name
