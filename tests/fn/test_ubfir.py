"""Test ubfir."""
import numpy as np
import pytest
from moirais.fn.ubfir import ubfir


def test_ubfir_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubfir(population=pop, area=area, n=20)
    assert r.value is not None


def test_ubfir_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubfir(population=pop, area=area, n=20)
    assert r.name
