"""Test ubfgm."""
import numpy as np
import pytest
from morie.fn.ubfgm import ubfgm


def test_ubfgm_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubfgm(population=pop, area=area, n=20)
    assert r.value is not None


def test_ubfgm_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubfgm(population=pop, area=area, n=20)
    assert r.name
