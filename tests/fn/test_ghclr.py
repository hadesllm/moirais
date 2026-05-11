"""Test ghclr."""
import numpy as np
import pytest
from morie.fn.ghclr import ghclr


def test_ghclr_basic():
    rng = np.random.default_rng(42)
    cases = rng.poisson(10, 20)
    controls = rng.poisson(100, 20) + 10
    exposure = rng.uniform(0, 1, 20)
    r = ghclr(cases=cases, controls=controls, exposure=exposure, n=20)
    assert r.value is not None


def test_ghclr_description():
    rng = np.random.default_rng(42)
    cases = rng.poisson(10, 20)
    controls = rng.poisson(100, 20) + 10
    exposure = rng.uniform(0, 1, 20)
    r = ghclr(cases=cases, controls=controls, exposure=exposure, n=20)
    assert r.name
