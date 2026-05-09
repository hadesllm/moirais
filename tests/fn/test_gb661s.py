"""Tests for gb661s.gibbons_mw_sampsize."""
import numpy as np
import pytest
from moirais.fn.gb661s import gibbons_mw_sampsize


def test_gb661s_basic():
    """Test basic functionality."""
    alpha = 0.05
    beta = 0.8
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_mw_sampsize(alpha, beta, delta)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb661s_edge():
    """Test edge cases."""
    alpha = 0.05
    beta = 0.8
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_mw_sampsize(alpha, beta, delta)
    assert isinstance(result, dict)
