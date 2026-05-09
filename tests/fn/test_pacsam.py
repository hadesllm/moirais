"""Tests for pacsam.sample_partial_autocorr."""
import numpy as np
import pytest
from moirais.fn.pacsam import sample_partial_autocorr


def test_pacsam_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    max_lag = np.random.default_rng(42).normal(0, 1, 100)
    result = sample_partial_autocorr(y, max_lag)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_pacsam_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    max_lag = np.random.default_rng(42).normal(0, 1, 100)
    result = sample_partial_autocorr(y, max_lag)
    assert isinstance(result, dict)
