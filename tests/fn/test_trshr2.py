"""Test trshr2."""
import numpy as np
import pytest
from morie.fn.trshr2 import trshr2


def test_trshr2_basic():
    rng = np.random.default_rng(42)
    flow = rng.poisson(500, 20)
    tt = rng.uniform(5, 60, 20)
    r = trshr2(flow_volume=flow, travel_time=tt, n=20)
    assert r.value is not None


def test_trshr2_description():
    rng = np.random.default_rng(42)
    flow = rng.poisson(500, 20)
    tt = rng.uniform(5, 60, 20)
    r = trshr2(flow_volume=flow, travel_time=tt, n=20)
    assert r.name
