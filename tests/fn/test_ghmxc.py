"""Test ghmxc."""
import numpy as np
import pytest
from morie.fn.ghmxc import ghmxc


def test_ghmxc_basic():
    rng = np.random.default_rng(42)
    cases = rng.poisson(10, 20)
    controls = rng.poisson(100, 20) + 10
    exposure = rng.uniform(0, 1, 20)
    r = ghmxc(cases=cases, controls=controls, exposure=exposure, n=20)
    assert r.value is not None


def test_ghmxc_description():
    rng = np.random.default_rng(42)
    cases = rng.poisson(10, 20)
    controls = rng.poisson(100, 20) + 10
    exposure = rng.uniform(0, 1, 20)
    r = ghmxc(cases=cases, controls=controls, exposure=exposure, n=20)
    assert r.name
