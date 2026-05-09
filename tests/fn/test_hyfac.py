"""Test hyfac."""
import numpy as np
import pytest
from moirais.fn.hyfac import hyfac


def test_hyfac_basic():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hyfac(flow=flow, precip=precip, n=20)
    assert r.value is not None


def test_hyfac_description():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hyfac(flow=flow, precip=precip, n=20)
    assert r.name
