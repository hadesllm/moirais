"""Test gdswg."""
import numpy as np
import pytest
from morie.fn.gdswg import gdswg


def test_gdswg_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(10000, 20)
    births = rng.poisson(100, 20)
    deaths = rng.poisson(80, 20)
    r = gdswg(population=pop, births=births, deaths=deaths, n=20)
    assert r.value is not None


def test_gdswg_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(10000, 20)
    births = rng.poisson(100, 20)
    deaths = rng.poisson(80, 20)
    r = gdswg(population=pop, births=births, deaths=deaths, n=20)
    assert r.name
