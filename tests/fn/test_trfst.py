"""Test trfst."""
import numpy as np
import pytest
from morie.fn.trfst import trfst


def test_trfst_basic():
    rng = np.random.default_rng(42)
    flow = rng.poisson(500, 20)
    tt = rng.uniform(5, 60, 20)
    r = trfst(flow_volume=flow, travel_time=tt, n=20)
    assert r.value is not None


def test_trfst_description():
    rng = np.random.default_rng(42)
    flow = rng.poisson(500, 20)
    tt = rng.uniform(5, 60, 20)
    r = trfst(flow_volume=flow, travel_time=tt, n=20)
    assert r.name
