"""Test hyfcr."""
import numpy as np
import pytest
from morie.fn.hyfcr import hyfcr


def test_hyfcr_basic():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hyfcr(flow=flow, precip=precip, n=20)
    assert r.value is not None


def test_hyfcr_description():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hyfcr(flow=flow, precip=precip, n=20)
    assert r.name
