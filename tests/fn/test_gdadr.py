"""Test gdadr."""
import numpy as np
import pytest
from moirais.fn.gdadr import gdadr


def test_gdadr_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(10000, 20)
    births = rng.poisson(100, 20)
    deaths = rng.poisson(80, 20)
    r = gdadr(population=pop, births=births, deaths=deaths, n=20)
    assert r.value is not None


def test_gdadr_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(10000, 20)
    births = rng.poisson(100, 20)
    deaths = rng.poisson(80, 20)
    r = gdadr(population=pop, births=births, deaths=deaths, n=20)
    assert r.name
