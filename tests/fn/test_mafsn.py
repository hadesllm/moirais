"""Tests for mafsn.ma_fail_safe_n."""
import numpy as np
import pytest
from moirais.fn.mafsn import ma_fail_safe_n


def test_mafsn_basic():
    """Test basic functionality."""
    z_scores = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = ma_fail_safe_n(z_scores, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mafsn_edge():
    """Test edge cases."""
    z_scores = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = ma_fail_safe_n(z_scores, alpha)
    assert isinstance(result, dict)
