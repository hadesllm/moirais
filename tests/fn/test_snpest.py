"""Tests for snpest.sn_pseudo_estimate."""
import numpy as np
import pytest
from moirais.fn.snpest import sn_pseudo_estimate


def test_snpest_basic():
    """Test basic functionality."""
    y_stream = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = sn_pseudo_estimate(y_stream, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_snpest_edge():
    """Test edge cases."""
    y_stream = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = sn_pseudo_estimate(y_stream, alpha)
    assert isinstance(result, dict)
