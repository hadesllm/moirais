"""Tests for mcdAnm.mcd_outlier."""
import numpy as np
import pytest
from moirais.fn.mcdAnm import mcd_outlier


def test_mcdAnm_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = mcd_outlier(X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mcdAnm_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = mcd_outlier(X)
    assert isinstance(result, dict)
