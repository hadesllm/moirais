"""Test ubcmp."""
import numpy as np
import pytest
from moirais.fn.ubcmp import ubcmp


def test_ubcmp_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubcmp(population=pop, area=area, n=20)
    assert r.value is not None


def test_ubcmp_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubcmp(population=pop, area=area, n=20)
    assert r.name
