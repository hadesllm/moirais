"""Test trbkl."""
import numpy as np
import pytest
from morie.fn.trbkl import trbkl


def test_trbkl_basic():
    rng = np.random.default_rng(42)
    flow = rng.poisson(500, 20)
    tt = rng.uniform(5, 60, 20)
    r = trbkl(flow_volume=flow, travel_time=tt, n=20)
    assert r.value is not None


def test_trbkl_description():
    rng = np.random.default_rng(42)
    flow = rng.poisson(500, 20)
    tt = rng.uniform(5, 60, 20)
    r = trbkl(flow_volume=flow, travel_time=tt, n=20)
    assert r.name
