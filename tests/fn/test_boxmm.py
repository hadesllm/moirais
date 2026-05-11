"""Test boxmm."""
import numpy as np
import pytest
from morie.fn.boxmm import box_m_test


def test_boxmm_basic():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 3))
    groups = np.array([0] * 30 + [1] * 30)
    r = box_m_test(X, groups)
    assert r.test_name == "Box's M"
    assert r.p_value >= 0


def test_boxmm_unequal():
    rng = np.random.default_rng(7)
    X1 = rng.standard_normal((30, 2))
    X2 = rng.standard_normal((30, 2)) * 3
    X = np.vstack([X1, X2])
    groups = np.array([0] * 30 + [1] * 30)
    r = box_m_test(X, groups)
    assert r.extra["n_groups"] == 2
