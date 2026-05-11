"""Test gdmed."""
import numpy as np
import pytest
from morie.fn.gdmed import gdmed


def test_gdmed_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(10000, 20)
    births = rng.poisson(100, 20)
    deaths = rng.poisson(80, 20)
    r = gdmed(population=pop, births=births, deaths=deaths, n=20)
    assert r.value is not None


def test_gdmed_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(10000, 20)
    births = rng.poisson(100, 20)
    deaths = rng.poisson(80, 20)
    r = gdmed(population=pop, births=births, deaths=deaths, n=20)
    assert r.name
