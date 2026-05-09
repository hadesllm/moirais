"""Test ubprk."""
import numpy as np
import pytest
from moirais.fn.ubprk import ubprk


def test_ubprk_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubprk(population=pop, area=area, n=20)
    assert r.value is not None


def test_ubprk_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubprk(population=pop, area=area, n=20)
    assert r.name
