"""Test trcrp."""
import numpy as np
import pytest
from moirais.fn.trcrp import trcrp


def test_trcrp_basic():
    rng = np.random.default_rng(42)
    flow = rng.poisson(500, 20)
    tt = rng.uniform(5, 60, 20)
    r = trcrp(flow_volume=flow, travel_time=tt, n=20)
    assert r.value is not None


def test_trcrp_description():
    rng = np.random.default_rng(42)
    flow = rng.poisson(500, 20)
    tt = rng.uniform(5, 60, 20)
    r = trcrp(flow_volume=flow, travel_time=tt, n=20)
    assert r.name
