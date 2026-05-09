"""Test gdhhs."""
import numpy as np
import pytest
from moirais.fn.gdhhs import gdhhs


def test_gdhhs_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(10000, 20)
    births = rng.poisson(100, 20)
    deaths = rng.poisson(80, 20)
    r = gdhhs(population=pop, births=births, deaths=deaths, n=20)
    assert r.value is not None


def test_gdhhs_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(10000, 20)
    births = rng.poisson(100, 20)
    deaths = rng.poisson(80, 20)
    r = gdhhs(population=pop, births=births, deaths=deaths, n=20)
    assert r.name
