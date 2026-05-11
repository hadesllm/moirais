"""Test ghscn."""
import numpy as np
import pytest
from morie.fn.ghscn import ghscn


def test_ghscn_basic():
    rng = np.random.default_rng(42)
    cases = rng.poisson(10, 20)
    controls = rng.poisson(100, 20) + 10
    exposure = rng.uniform(0, 1, 20)
    r = ghscn(cases=cases, controls=controls, exposure=exposure, n=20)
    assert r.value is not None


def test_ghscn_description():
    rng = np.random.default_rng(42)
    cases = rng.poisson(10, 20)
    controls = rng.poisson(100, 20) + 10
    exposure = rng.uniform(0, 1, 20)
    r = ghscn(cases=cases, controls=controls, exposure=exposure, n=20)
    assert r.name
