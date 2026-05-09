"""Test hynit."""
import numpy as np
import pytest
from moirais.fn.hynit import hynit


def test_hynit_basic():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hynit(flow=flow, precip=precip, n=20)
    assert r.value is not None


def test_hynit_description():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hynit(flow=flow, precip=precip, n=20)
    assert r.name
