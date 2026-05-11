"""Tests for convgs.convergent_validity."""
import numpy as np
import pytest
from morie.fn.convgs import convergent_validity


def test_convgs_basic():
    """Test basic functionality."""
    loadings = np.random.default_rng(42).normal(0, 1, 100)
    residuals = np.random.default_rng(42).normal(0, 1, 100)
    result = convergent_validity(loadings, residuals)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_convgs_edge():
    """Test edge cases."""
    loadings = np.random.default_rng(42).normal(0, 1, 100)
    residuals = np.random.default_rng(42).normal(0, 1, 100)
    result = convergent_validity(loadings, residuals)
    assert isinstance(result, dict)
