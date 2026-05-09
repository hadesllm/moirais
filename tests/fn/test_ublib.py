"""Test ublib."""
import numpy as np
import pytest
from moirais.fn.ublib import ublib


def test_ublib_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ublib(population=pop, area=area, n=20)
    assert r.value is not None


def test_ublib_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ublib(population=pop, area=area, n=20)
    assert r.name
