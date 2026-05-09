"""Test ubbke."""
import numpy as np
import pytest
from moirais.fn.ubbke import ubbke


def test_ubbke_basic():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubbke(population=pop, area=area, n=20)
    assert r.value is not None


def test_ubbke_description():
    rng = np.random.default_rng(42)
    pop = rng.poisson(5000, 20)
    area = rng.uniform(1, 100, 20)
    r = ubbke(population=pop, area=area, n=20)
    assert r.name
