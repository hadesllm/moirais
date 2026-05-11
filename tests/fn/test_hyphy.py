"""Test hyphy."""
import numpy as np
import pytest
from morie.fn.hyphy import hyphy


def test_hyphy_basic():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hyphy(flow=flow, precip=precip, n=20)
    assert isinstance(r.value, float) and np.isfinite(r.value)
    assert r.value > 0
    assert r.value == pytest.approx(np.mean(flow), rel=1e-10)


def test_hyphy_description():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hyphy(flow=flow, precip=precip, n=20)
    assert isinstance(r.name, str) and len(r.name) > 0
