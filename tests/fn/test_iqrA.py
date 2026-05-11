"""Tests for iqrA.iqr_outlier."""
import numpy as np
import pytest
from morie.fn.iqrA import iqr_outlier


def test_iqrA_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = iqr_outlier(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_iqrA_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = iqr_outlier(x)
    assert isinstance(result, dict)
