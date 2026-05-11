"""Test hycti."""
import numpy as np
import pytest
from morie.fn.hycti import hycti


def test_hycti_basic():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hycti(flow=flow, precip=precip, n=20)
    assert r.value is not None


def test_hycti_description():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hycti(flow=flow, precip=precip, n=20)
    assert r.name
