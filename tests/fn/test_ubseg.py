"""Test ubseg."""
import numpy as np
import pytest
from morie.fn.ubseg import ubseg


def test_ubseg_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubseg(population=pop, area=area, n=20)
    assert r.value is not None


def test_ubseg_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubseg(population=pop, area=area, n=20)
    assert r.name
