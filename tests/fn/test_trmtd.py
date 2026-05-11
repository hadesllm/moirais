"""Test trmtd."""
import numpy as np
import pytest
from morie.fn.trmtd import trmtd


def test_trmtd_basic():
    rng = np.random.default_rng(42)
    flow = rng.poisson(500, 20)
    tt = rng.uniform(5, 60, 20)
    r = trmtd(flow_volume=flow, travel_time=tt, n=20)
    assert r.value is not None


def test_trmtd_description():
    rng = np.random.default_rng(42)
    flow = rng.poisson(500, 20)
    tt = rng.uniform(5, 60, 20)
    r = trmtd(flow_volume=flow, travel_time=tt, n=20)
    assert r.name
