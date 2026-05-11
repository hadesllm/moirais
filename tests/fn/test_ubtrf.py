"""Test ubtrf."""
import numpy as np
import pytest
from morie.fn.ubtrf import ubtrf


def test_ubtrf_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubtrf(population=pop, area=area, n=20)
    assert r.value is not None


def test_ubtrf_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubtrf(population=pop, area=area, n=20)
    assert r.name
