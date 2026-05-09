"""Test gdntn."""
import numpy as np
import pytest
from moirais.fn.gdntn import gdntn


def test_gdntn_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(10000, 20)
    births = rng.poisson(100, 20)
    deaths = rng.poisson(80, 20)
    r = gdntn(population=pop, births=births, deaths=deaths, n=20)
    assert r.value is not None


def test_gdntn_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(10000, 20)
    births = rng.poisson(100, 20)
    deaths = rng.poisson(80, 20)
    r = gdntn(population=pop, births=births, deaths=deaths, n=20)
    assert r.name
