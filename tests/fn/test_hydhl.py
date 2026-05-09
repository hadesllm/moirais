"""Test hydhl."""
import numpy as np
import pytest
from moirais.fn.hydhl import hydhl


def test_hydhl_basic():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hydhl(flow=flow, precip=precip, n=20)
    assert r.value is not None


def test_hydhl_description():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hydhl(flow=flow, precip=precip, n=20)
    assert r.name
