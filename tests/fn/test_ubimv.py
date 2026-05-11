"""Test ubimv."""
import numpy as np
import pytest
from morie.fn.ubimv import ubimv


def test_ubimv_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubimv(population=pop, area=area, n=20)
    assert r.value is not None


def test_ubimv_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubimv(population=pop, area=area, n=20)
    assert r.name
