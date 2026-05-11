"""Test ghacc."""
import numpy as np
import pytest
from morie.fn.ghacc import ghacc


def test_ghacc_basic():
    rng = np.random.default_rng(42)
    cases = rng.poisson(10, 20)
    controls = rng.poisson(100, 20) + 10
    exposure = rng.uniform(0, 1, 20)
    r = ghacc(cases=cases, controls=controls, exposure=exposure, n=20)
    assert r.value is not None


def test_ghacc_description():
    rng = np.random.default_rng(42)
    cases = rng.poisson(10, 20)
    controls = rng.poisson(100, 20) + 10
    exposure = rng.uniform(0, 1, 20)
    r = ghacc(cases=cases, controls=controls, exposure=exposure, n=20)
    assert r.name
