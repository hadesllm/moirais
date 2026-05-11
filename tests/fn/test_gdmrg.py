"""Test gdmrg."""
import numpy as np
import pytest
from morie.fn.gdmrg import gdmrg


def test_gdmrg_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(10000, 20)
    births = rng.poisson(100, 20)
    deaths = rng.poisson(80, 20)
    r = gdmrg(population=pop, births=births, deaths=deaths, n=20)
    assert r.value is not None


def test_gdmrg_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(10000, 20)
    births = rng.poisson(100, 20)
    deaths = rng.poisson(80, 20)
    r = gdmrg(population=pop, births=births, deaths=deaths, n=20)
    assert r.name
