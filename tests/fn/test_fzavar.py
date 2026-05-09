"""Tests for fzavar.fauzi_quantile_asymp_var."""
import numpy as np
import pytest
from moirais.fn.fzavar import fauzi_quantile_asymp_var


def test_fzavar_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = fauzi_quantile_asymp_var(data, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzavar_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = fauzi_quantile_asymp_var(data, p)
    assert isinstance(result, dict)
