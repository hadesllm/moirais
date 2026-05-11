"""Test truer."""
import numpy as np
import pytest
from morie.fn.truer import truer


def test_truer_basic():
    rng = np.random.default_rng(42)
    flow = rng.poisson(500, 20)
    tt = rng.uniform(5, 60, 20)
    r = truer(flow_volume=flow, travel_time=tt, n=20)
    assert r.value is not None


def test_truer_description():
    rng = np.random.default_rng(42)
    flow = rng.poisson(500, 20)
    tt = rng.uniform(5, 60, 20)
    r = truer(flow_volume=flow, travel_time=tt, n=20)
    assert r.name
