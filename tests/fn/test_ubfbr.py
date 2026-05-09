"""Test ubfbr."""
import numpy as np
import pytest
from moirais.fn.ubfbr import ubfbr


def test_ubfbr_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubfbr(population=pop, area=area, n=20)
    assert r.value is not None


def test_ubfbr_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubfbr(population=pop, area=area, n=20)
    assert r.name
