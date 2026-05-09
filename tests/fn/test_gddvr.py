"""Test gddvr."""
import numpy as np
import pytest
from moirais.fn.gddvr import gddvr


def test_gddvr_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(10000, 20)
    births = rng.poisson(100, 20)
    deaths = rng.poisson(80, 20)
    r = gddvr(population=pop, births=births, deaths=deaths, n=20)
    assert r.value is not None


def test_gddvr_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(10000, 20)
    births = rng.poisson(100, 20)
    deaths = rng.poisson(80, 20)
    r = gddvr(population=pop, births=births, deaths=deaths, n=20)
    assert r.name
