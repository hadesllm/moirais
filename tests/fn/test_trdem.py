"""Test trdem."""
import numpy as np
import pytest
from morie.fn.trdem import trdem


def test_trdem_basic():
    rng = np.random.default_rng(42)
    flow = rng.poisson(500, 20)
    tt = rng.uniform(5, 60, 20)
    r = trdem(flow_volume=flow, travel_time=tt, n=20)
    assert r.value is not None


def test_trdem_description():
    rng = np.random.default_rng(42)
    flow = rng.poisson(500, 20)
    tt = rng.uniform(5, 60, 20)
    r = trdem(flow_volume=flow, travel_time=tt, n=20)
    assert r.name
