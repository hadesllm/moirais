"""Tests for trcwgt.truncated_combined_weights."""
import numpy as np
import pytest
from moirais.fn.trcwgt import truncated_combined_weights


def test_trcwgt_basic():
    """Test basic functionality."""
    sw_A = np.random.default_rng(42).normal(0, 1, 100)
    sw_C = np.random.default_rng(42).normal(0, 1, 100)
    quantile = np.random.default_rng(42).normal(0, 1, 100)
    result = truncated_combined_weights(sw_A, sw_C, quantile)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_trcwgt_edge():
    """Test edge cases."""
    sw_A = np.random.default_rng(42).normal(0, 1, 100)
    sw_C = np.random.default_rng(42).normal(0, 1, 100)
    quantile = np.random.default_rng(42).normal(0, 1, 100)
    result = truncated_combined_weights(sw_A, sw_C, quantile)
    assert isinstance(result, dict)
