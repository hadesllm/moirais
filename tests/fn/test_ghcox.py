"""Test ghcox."""
import numpy as np
import pytest
from morie.fn.ghcox import ghcox


def test_ghcox_basic():
    rng = np.random.default_rng(42)
    cases = rng.poisson(10, 20)
    controls = rng.poisson(100, 20) + 10
    exposure = rng.uniform(0, 1, 20)
    r = ghcox(cases=cases, controls=controls, exposure=exposure, n=20)
    assert r.value is not None


def test_ghcox_description():
    rng = np.random.default_rng(42)
    cases = rng.poisson(10, 20)
    controls = rng.poisson(100, 20) + 10
    exposure = rng.uniform(0, 1, 20)
    r = ghcox(cases=cases, controls=controls, exposure=exposure, n=20)
    assert r.name
