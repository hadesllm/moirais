"""Test ghgap."""
import numpy as np
import pytest
from moirais.fn.ghgap import ghgap


def test_ghgap_basic():
    rng = np.random.default_rng(42)
    cases = rng.poisson(10, 20)
    controls = rng.poisson(100, 20) + 10
    exposure = rng.uniform(0, 1, 20)
    r = ghgap(cases=cases, controls=controls, exposure=exposure, n=20)
    assert r.value is not None


def test_ghgap_description():
    rng = np.random.default_rng(42)
    cases = rng.poisson(10, 20)
    controls = rng.poisson(100, 20) + 10
    exposure = rng.uniform(0, 1, 20)
    r = ghgap(cases=cases, controls=controls, exposure=exposure, n=20)
    assert r.name
