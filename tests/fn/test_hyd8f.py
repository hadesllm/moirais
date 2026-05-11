"""Test hyd8f."""
import numpy as np
import pytest
from morie.fn.hyd8f import hyd8f


def test_hyd8f_basic():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hyd8f(flow=flow, precip=precip, n=20)
    assert r.value is not None


def test_hyd8f_description():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hyd8f(flow=flow, precip=precip, n=20)
    assert r.name
