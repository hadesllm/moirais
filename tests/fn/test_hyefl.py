"""Test hyefl."""
import numpy as np
import pytest
from moirais.fn.hyefl import hyefl


def test_hyefl_basic():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hyefl(flow=flow, precip=precip, n=20)
    assert r.value is not None


def test_hyefl_description():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hyefl(flow=flow, precip=precip, n=20)
    assert r.name
