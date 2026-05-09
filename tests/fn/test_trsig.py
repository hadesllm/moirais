"""Test trsig."""
import numpy as np
import pytest
from moirais.fn.trsig import trsig


def test_trsig_basic():
    rng = np.random.default_rng(42)
    flow = rng.poisson(500, 20)
    tt = rng.uniform(5, 60, 20)
    r = trsig(flow_volume=flow, travel_time=tt, n=20)
    assert r.value is not None


def test_trsig_description():
    rng = np.random.default_rng(42)
    flow = rng.poisson(500, 20)
    tt = rng.uniform(5, 60, 20)
    r = trsig(flow_volume=flow, travel_time=tt, n=20)
    assert r.name
