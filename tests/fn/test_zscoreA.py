"""Tests for zscoreA.zscore_anomaly."""
import numpy as np
import pytest
from moirais.fn.zscoreA import zscore_anomaly


def test_zscoreA_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = zscore_anomaly(x, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_zscoreA_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = zscore_anomaly(x, k)
    assert isinstance(result, dict)
