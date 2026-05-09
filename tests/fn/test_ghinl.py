"""Test ghinl."""
import numpy as np
import pytest
from moirais.fn.ghinl import ghinl


def test_ghinl_basic():
    rng = np.random.default_rng(42)
    cases = rng.poisson(10, 20)
    controls = rng.poisson(100, 20) + 10
    exposure = rng.uniform(0, 1, 20)
    r = ghinl(cases=cases, controls=controls, exposure=exposure, n=20)
    assert r.value is not None


def test_ghinl_description():
    rng = np.random.default_rng(42)
    cases = rng.poisson(10, 20)
    controls = rng.poisson(100, 20) + 10
    exposure = rng.uniform(0, 1, 20)
    r = ghinl(cases=cases, controls=controls, exposure=exposure, n=20)
    assert r.name
