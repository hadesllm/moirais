"""Test hywtq."""
import numpy as np
import pytest
from moirais.fn.hywtq import hywtq


def test_hywtq_basic():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hywtq(flow=flow, precip=precip, n=20)
    assert r.value is not None


def test_hywtq_description():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hywtq(flow=flow, precip=precip, n=20)
    assert r.name
