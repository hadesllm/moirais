"""Test trsgn."""
import numpy as np
import pytest
from moirais.fn.trsgn import trsgn


def test_trsgn_basic():
    rng = np.random.default_rng(42)
    flow = rng.poisson(500, 20)
    tt = rng.uniform(5, 60, 20)
    r = trsgn(flow_volume=flow, travel_time=tt, n=20)
    assert r.value is not None


def test_trsgn_description():
    rng = np.random.default_rng(42)
    flow = rng.poisson(500, 20)
    tt = rng.uniform(5, 60, 20)
    r = trsgn(flow_volume=flow, travel_time=tt, n=20)
    assert r.name
