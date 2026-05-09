"""Test gdmml."""
import numpy as np
import pytest
from moirais.fn.gdmml import gdmml


def test_gdmml_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(10000, 20)
    births = rng.poisson(100, 20)
    deaths = rng.poisson(80, 20)
    r = gdmml(population=pop, births=births, deaths=deaths, n=20)
    assert r.value is not None


def test_gdmml_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(10000, 20)
    births = rng.poisson(100, 20)
    deaths = rng.poisson(80, 20)
    r = gdmml(population=pop, births=births, deaths=deaths, n=20)
    assert r.name
