"""Test ghfca."""
import numpy as np
import pytest
from moirais.fn.ghfca import ghfca


def test_ghfca_basic():
    rng = np.random.default_rng(42)
    cases = rng.poisson(10, 20)
    controls = rng.poisson(100, 20) + 10
    exposure = rng.uniform(0, 1, 20)
    r = ghfca(cases=cases, controls=controls, exposure=exposure, n=20)
    assert r.value is not None


def test_ghfca_description():
    rng = np.random.default_rng(42)
    cases = rng.poisson(10, 20)
    controls = rng.poisson(100, 20) + 10
    exposure = rng.uniform(0, 1, 20)
    r = ghfca(cases=cases, controls=controls, exposure=exposure, n=20)
    assert r.name
