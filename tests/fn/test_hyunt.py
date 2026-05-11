"""Test hyunt."""
import numpy as np
import pytest
from morie.fn.hyunt import hyunt


def test_hyunt_basic():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hyunt(flow=flow, precip=precip, n=20)
    assert r.value is not None


def test_hyunt_description():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hyunt(flow=flow, precip=precip, n=20)
    assert r.name
