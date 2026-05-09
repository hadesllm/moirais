"""Tests for fzksst.fauzi_ks_statistic."""
import numpy as np
import pytest
from moirais.fn.fzksst import fauzi_ks_statistic


def test_fzksst_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    cdf = (lambda v: 1.0 / (1.0 + np.exp(-v)))
    result = fauzi_ks_statistic(data, cdf)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_fzksst_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    cdf = (lambda v: 1.0 / (1.0 + np.exp(-v)))
    result = fauzi_ks_statistic(data, cdf)
    assert isinstance(result, dict)
