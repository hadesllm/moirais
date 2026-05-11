"""Tests for replwt.replicate_weights."""
import numpy as np
import pytest
from morie.fn.replwt import replicate_weights


def test_replwt_basic():
    """Test basic functionality."""
    design = np.random.default_rng(42).normal(0, 1, 100)
    method = 'auto'
    R = np.random.default_rng(42).normal(0, 1, 100)
    result = replicate_weights(design, method, R)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_replwt_edge():
    """Test edge cases."""
    design = np.random.default_rng(42).normal(0, 1, 100)
    method = 'auto'
    R = np.random.default_rng(42).normal(0, 1, 100)
    result = replicate_weights(design, method, R)
    assert isinstance(result, dict)
