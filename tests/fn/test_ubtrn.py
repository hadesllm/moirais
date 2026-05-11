"""Test ubtrn."""
import numpy as np
import pytest
from morie.fn.ubtrn import ubtrn


def test_ubtrn_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubtrn(population=pop, area=area, n=20)
    assert r.value is not None


def test_ubtrn_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubtrn(population=pop, area=area, n=20)
    assert r.name
