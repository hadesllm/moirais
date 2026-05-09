"""Test trprt."""
import numpy as np
import pytest
from moirais.fn.trprt import trprt


def test_trprt_basic():
    rng = np.random.default_rng(42)
    flow = rng.poisson(500, 20)
    tt = rng.uniform(5, 60, 20)
    r = trprt(flow_volume=flow, travel_time=tt, n=20)
    assert r.value is not None


def test_trprt_description():
    rng = np.random.default_rng(42)
    flow = rng.poisson(500, 20)
    tt = rng.uniform(5, 60, 20)
    r = trprt(flow_volume=flow, travel_time=tt, n=20)
    assert r.name
