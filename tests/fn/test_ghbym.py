"""Test ghbym."""
import numpy as np
import pytest
from moirais.fn.ghbym import ghbym


def test_ghbym_basic():
    rng = np.random.default_rng(42)
    cases = rng.poisson(10, 20)
    controls = rng.poisson(100, 20) + 10
    exposure = rng.uniform(0, 1, 20)
    r = ghbym(cases=cases, controls=controls, exposure=exposure, n=20)
    assert r.value is not None


def test_ghbym_description():
    rng = np.random.default_rng(42)
    cases = rng.poisson(10, 20)
    controls = rng.poisson(100, 20) + 10
    exposure = rng.uniform(0, 1, 20)
    r = ghbym(cases=cases, controls=controls, exposure=exposure, n=20)
    assert r.name
