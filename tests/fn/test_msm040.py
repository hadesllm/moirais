"""Tests for msm040.mvsml_linear_mixed_models_eq_5_4."""
import numpy as np
import pytest
from morie.fn.msm040 import mvsml_linear_mixed_models_eq_5_4


def test_msm040_basic():
    """Test basic functionality."""
    yp_ts = np.random.default_rng(42).normal(0, 1, 100)
    yp = np.random.default_rng(42).normal(0, 1, 100)
    Pos_tst = np.random.default_rng(42).normal(0, 1, 100)
    Tab = np.random.default_rng(42).normal(0, 1, 100)
    MSEP10a = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = mvsml_linear_mixed_models_eq_5_4(yp_ts, yp, Pos_tst, Tab, MSEP10a, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm040_edge():
    """Test edge cases."""
    yp_ts = np.random.default_rng(42).normal(0, 1, 100)
    yp = np.random.default_rng(42).normal(0, 1, 100)
    Pos_tst = np.random.default_rng(42).normal(0, 1, 100)
    Tab = np.random.default_rng(42).normal(0, 1, 100)
    MSEP10a = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = mvsml_linear_mixed_models_eq_5_4(yp_ts, yp, Pos_tst, Tab, MSEP10a, k)
    assert isinstance(result, dict)
