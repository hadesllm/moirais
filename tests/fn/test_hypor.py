"""Test hypor."""
import numpy as np
import pytest
from moirais.fn.hypor import hypor


def test_hypor_basic():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hypor(flow=flow, precip=precip, n=20)
    assert isinstance(r.value, float) and np.isfinite(r.value)
    assert r.value > 0
    assert r.value == pytest.approx(np.mean(flow), rel=1e-10)


def test_hypor_description():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hypor(flow=flow, precip=precip, n=20)
    assert isinstance(r.name, str) and len(r.name) > 0
