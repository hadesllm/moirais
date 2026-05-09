"""Test hysnk."""
import numpy as np
import pytest
from moirais.fn.hysnk import hysnk


def test_hysnk_basic():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hysnk(flow=flow, precip=precip, n=20)
    assert r.value is not None


def test_hysnk_description():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hysnk(flow=flow, precip=precip, n=20)
    assert r.name
