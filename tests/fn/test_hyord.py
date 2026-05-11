"""Test hyord."""
import numpy as np
import pytest
from morie.fn.hyord import hyord


def test_hyord_basic():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hyord(flow=flow, precip=precip, n=20)
    assert r.value is not None


def test_hyord_description():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hyord(flow=flow, precip=precip, n=20)
    assert r.name
