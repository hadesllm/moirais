"""Test bfdr."""
import numpy as np
import pytest
from morie.fn.bfdr import bayesian_fdr


def test_bfdr_basic():
    pp = np.array([0.9, 0.8, 0.3, 0.1, 0.95])
    r = bayesian_fdr(pp, threshold=0.5)
    assert 0.0 <= r.value <= 1.0
    assert r.extra["n_discoveries"] == 3


def test_bfdr_no_discoveries():
    pp = np.array([0.1, 0.2, 0.3])
    r = bayesian_fdr(pp, threshold=0.9)
    assert r.value == 0.0
    assert r.extra["n_discoveries"] == 0
