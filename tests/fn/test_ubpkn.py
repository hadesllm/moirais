"""Test ubpkn."""
import numpy as np
import pytest
from moirais.fn.ubpkn import ubpkn


def test_ubpkn_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubpkn(population=pop, area=area, n=20)
    assert r.value is not None


def test_ubpkn_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubpkn(population=pop, area=area, n=20)
    assert r.name
