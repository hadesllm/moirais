"""Tests for tldepu.upper_tail_dependence."""
import numpy as np
import pytest
from moirais.fn.tldepu import upper_tail_dependence


def test_tldepu_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    copula = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = upper_tail_dependence(y, copula, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tldepu_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    copula = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = upper_tail_dependence(y, copula, theta)
    assert isinstance(result, dict)
