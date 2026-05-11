"""Test hyshr."""
import numpy as np
import pytest
from morie.fn.hyshr import hyshr


def test_hyshr_basic():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hyshr(flow=flow, precip=precip, n=20)
    assert r.value is not None


def test_hyshr_description():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hyshr(flow=flow, precip=precip, n=20)
    assert r.name
