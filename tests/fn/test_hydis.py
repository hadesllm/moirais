"""Test hydis."""
import numpy as np
import pytest
from morie.fn.hydis import hydis


def test_hydis_basic():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hydis(flow=flow, precip=precip, n=20)
    assert r.value is not None


def test_hydis_description():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hydis(flow=flow, precip=precip, n=20)
    assert r.name
