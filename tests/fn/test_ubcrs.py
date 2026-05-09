"""Test ubcrs."""
import numpy as np
import pytest
from moirais.fn.ubcrs import ubcrs


def test_ubcrs_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubcrs(population=pop, area=area, n=20)
    assert r.value is not None


def test_ubcrs_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubcrs(population=pop, area=area, n=20)
    assert r.name
