"""Test hyaqu."""
import numpy as np
import pytest
from morie.fn.hyaqu import hyaqu


def test_hyaqu_basic():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hyaqu(flow=flow, precip=precip, n=20)
    assert r.value is not None


def test_hyaqu_description():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hyaqu(flow=flow, precip=precip, n=20)
    assert r.name
