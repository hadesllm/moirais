"""Test hyevp."""
import numpy as np
import pytest
from morie.fn.hyevp import hyevp


def test_hyevp_basic():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hyevp(flow=flow, precip=precip, n=20)
    assert r.value is not None


def test_hyevp_description():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hyevp(flow=flow, precip=precip, n=20)
    assert r.name
