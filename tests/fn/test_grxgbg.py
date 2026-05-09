"""Tests for grxgbg.geron_xgboost_gain."""
import numpy as np
import pytest
from moirais.fn.grxgbg import geron_xgboost_gain


def test_grxgbg_basic():
    """Test basic functionality."""
    GL = np.random.default_rng(42).normal(0, 1, 100)
    HL = np.random.default_rng(42).normal(0, 1, 100)
    GR = np.random.default_rng(42).normal(0, 1, 100)
    HR = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    gamma = 1.0
    result = geron_xgboost_gain(GL, HL, GR, HR, lam, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grxgbg_edge():
    """Test edge cases."""
    GL = np.random.default_rng(42).normal(0, 1, 100)
    HL = np.random.default_rng(42).normal(0, 1, 100)
    GR = np.random.default_rng(42).normal(0, 1, 100)
    HR = np.random.default_rng(42).normal(0, 1, 100)
    lam = 0.1
    gamma = 1.0
    result = geron_xgboost_gain(GL, HL, GR, HR, lam, gamma)
    assert isinstance(result, dict)
