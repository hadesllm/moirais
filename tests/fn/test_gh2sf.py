"""Test gh2sf."""
import numpy as np
import pytest
from moirais.fn.gh2sf import gh2sf


def test_gh2sf_basic():
    rng = np.random.default_rng(42)
    cases = rng.poisson(10, 20)
    controls = rng.poisson(100, 20) + 10
    exposure = rng.uniform(0, 1, 20)
    r = gh2sf(cases=cases, controls=controls, exposure=exposure, n=20)
    assert r.value is not None


def test_gh2sf_description():
    rng = np.random.default_rng(42)
    cases = rng.poisson(10, 20)
    controls = rng.poisson(100, 20) + 10
    exposure = rng.uniform(0, 1, 20)
    r = gh2sf(cases=cases, controls=controls, exposure=exposure, n=20)
    assert r.name
