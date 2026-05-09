"""Test ubdis."""
import numpy as np
import pytest
from moirais.fn.ubdis import ubdis


def test_ubdis_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubdis(population=pop, area=area, n=20)
    assert r.value is not None


def test_ubdis_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubdis(population=pop, area=area, n=20)
    assert r.name
