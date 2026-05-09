"""Test hydfl."""
import numpy as np
import pytest
from moirais.fn.hydfl import hydfl


def test_hydfl_basic():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hydfl(flow=flow, precip=precip, n=20)
    assert r.value is not None


def test_hydfl_description():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hydfl(flow=flow, precip=precip, n=20)
    assert r.name
