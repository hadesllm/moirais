"""Test hyret."""
import numpy as np
import pytest
from morie.fn.hyret import hyret


def test_hyret_basic():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hyret(flow=flow, precip=precip, n=20)
    assert r.value is not None


def test_hyret_description():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hyret(flow=flow, precip=precip, n=20)
    assert r.name
