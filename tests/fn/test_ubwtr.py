"""Test ubwtr."""
import numpy as np
import pytest
from moirais.fn.ubwtr import ubwtr


def test_ubwtr_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubwtr(population=pop, area=area, n=20)
    assert r.value is not None


def test_ubwtr_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubwtr(population=pop, area=area, n=20)
    assert r.name
