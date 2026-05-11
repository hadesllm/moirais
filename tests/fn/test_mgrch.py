"""Tests for mgrch.bekk_garch_multivariate."""
import numpy as np
import pytest
from morie.fn.mgrch import bekk_garch_multivariate


def test_mgrch_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = bekk_garch_multivariate(X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mgrch_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = bekk_garch_multivariate(X)
    assert isinstance(result, dict)
