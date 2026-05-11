"""Test ubcls."""
import numpy as np
import pytest
from morie.fn.ubcls import ubcls


def test_ubcls_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubcls(population=pop, area=area, n=20)
    assert r.value is not None


def test_ubcls_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubcls(population=pop, area=area, n=20)
    assert r.name
