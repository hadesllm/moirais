"""Test gdtnd."""
import numpy as np
import pytest
from morie.fn.gdtnd import gdtnd


def test_gdtnd_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(10000, 20)
    births = rng.poisson(100, 20)
    deaths = rng.poisson(80, 20)
    r = gdtnd(population=pop, births=births, deaths=deaths, n=20)
    assert r.value is not None


def test_gdtnd_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(10000, 20)
    births = rng.poisson(100, 20)
    deaths = rng.poisson(80, 20)
    r = gdtnd(population=pop, births=births, deaths=deaths, n=20)
    assert r.name
