"""Test ghsem."""
import numpy as np
import pytest
from moirais.fn.ghsem import ghsem


def test_ghsem_basic():
    rng = np.random.default_rng(42)
    cases = rng.poisson(10, 20)
    controls = rng.poisson(100, 20) + 10
    exposure = rng.uniform(0, 1, 20)
    r = ghsem(cases=cases, controls=controls, exposure=exposure, n=20)
    assert r.value is not None


def test_ghsem_description():
    rng = np.random.default_rng(42)
    cases = rng.poisson(10, 20)
    controls = rng.poisson(100, 20) + 10
    exposure = rng.uniform(0, 1, 20)
    r = ghsem(cases=cases, controls=controls, exposure=exposure, n=20)
    assert r.name
