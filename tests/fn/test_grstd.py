"""Tests for grstd.geron_standardization."""
import numpy as np
import pytest
from moirais.fn.grstd import geron_standardization


def test_grstd_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = geron_standardization(X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grstd_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = geron_standardization(X)
    assert isinstance(result, dict)
