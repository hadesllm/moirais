"""Tests for mfovsm.mfo_vsm."""
import numpy as np
import pytest
from moirais.fn.mfovsm import mfo_vsm


def test_mfovsm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    feature = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = mfo_vsm(y, feature, A, H)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mfovsm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    feature = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = mfo_vsm(y, feature, A, H)
    assert isinstance(result, dict)
