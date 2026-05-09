"""Test csrac."""
import numpy as np
import pytest
from moirais.fn.csrac import csrac


def test_csrac_basic():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = csrac(incidents=inc, population=pop, n=20)
    assert r.value is not None


def test_csrac_description():
    rng = np.random.default_rng(42)
    inc = rng.poisson(20, 20)
    pop = rng.poisson(5000, 20) + 100
    r = csrac(incidents=inc, population=pop, n=20)
    assert r.name
