"""Test ubthr."""
import numpy as np
import pytest
from morie.fn.ubthr import ubthr


def test_ubthr_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubthr(population=pop, area=area, n=20)
    assert r.value is not None


def test_ubthr_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubthr(population=pop, area=area, n=20)
    assert r.name
