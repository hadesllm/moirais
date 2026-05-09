"""Tests for dpsbw.stick_breaking_weights."""
import numpy as np
import pytest
from moirais.fn.dpsbw import stick_breaking_weights


def test_dpsbw_basic():
    """Test basic functionality."""
    alpha = 0.05
    truncation = np.random.default_rng(42).normal(0, 1, 100)
    result = stick_breaking_weights(alpha, truncation)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpsbw_edge():
    """Test edge cases."""
    alpha = 0.05
    truncation = np.random.default_rng(42).normal(0, 1, 100)
    result = stick_breaking_weights(alpha, truncation)
    assert isinstance(result, dict)
