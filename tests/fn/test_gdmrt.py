"""Test gdmrt."""
import numpy as np
import pytest
from moirais.fn.gdmrt import gdmrt


def test_gdmrt_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(10000, 20)
    births = rng.poisson(100, 20)
    deaths = rng.poisson(80, 20)
    r = gdmrt(population=pop, births=births, deaths=deaths, n=20)
    assert r.value is not None


def test_gdmrt_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(10000, 20)
    births = rng.poisson(100, 20)
    deaths = rng.poisson(80, 20)
    r = gdmrt(population=pop, births=births, deaths=deaths, n=20)
    assert r.name
