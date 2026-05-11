"""Test qnest."""
import numpy as np
import pytest
from morie.fn.qnest import qn_estimator


def test_qnest_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = qn_estimator(x)
    assert r.value > 0
    assert r.name == "qnest"


def test_qnest_small():
    r = qn_estimator(np.array([1.0, 2.0, 3.0]))
    assert r.value >= 0
    assert r.extra["n"] == 3
