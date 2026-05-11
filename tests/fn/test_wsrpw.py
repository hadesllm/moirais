"""Tests for wsrpw.wilcoxon_power."""
import numpy as np
import pytest
from morie.fn.wsrpw import wilcoxon_power


def test_wsrpw_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcoxon_power(x)
    assert 'statistic' in result
    assert 'p_value' in result
    assert 0 <= result['p_value'] <= 1


def test_wsrpw_edge():
    """Test edge cases."""
    result = wilcoxon_power(np.array([1.0]))
    assert result['n'] == 1
