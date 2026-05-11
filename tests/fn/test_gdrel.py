"""Test gdrel."""
import numpy as np
import pytest
from morie.fn.gdrel import gdrel


def test_gdrel_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(10000, 20)
    births = rng.poisson(100, 20)
    deaths = rng.poisson(80, 20)
    r = gdrel(population=pop, births=births, deaths=deaths, n=20)
    assert r.value is not None


def test_gdrel_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(10000, 20)
    births = rng.poisson(100, 20)
    deaths = rng.poisson(80, 20)
    r = gdrel(population=pop, births=births, deaths=deaths, n=20)
    assert r.name
