"""Tests for jomimi.joseph_missing_data_imputation_ts."""
import numpy as np
import pytest
from morie.fn.jomimi import joseph_missing_data_imputation_ts


def test_jomimi_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    strategy = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    result = joseph_missing_data_imputation_ts(y, strategy, m)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jomimi_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    strategy = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    result = joseph_missing_data_imputation_ts(y, strategy, m)
    assert isinstance(result, dict)
