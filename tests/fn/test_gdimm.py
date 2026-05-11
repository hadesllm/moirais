"""Test gdimm."""
import numpy as np
import pytest
from morie.fn.gdimm import gdimm


def test_gdimm_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(10000, 20)
    births = rng.poisson(100, 20)
    deaths = rng.poisson(80, 20)
    r = gdimm(population=pop, births=births, deaths=deaths, n=20)
    assert r.value is not None


def test_gdimm_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(10000, 20)
    births = rng.poisson(100, 20)
    deaths = rng.poisson(80, 20)
    r = gdimm(population=pop, births=births, deaths=deaths, n=20)
    assert r.name
