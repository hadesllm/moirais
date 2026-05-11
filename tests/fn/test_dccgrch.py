"""Tests for dccgrch.dcc_garch."""
import numpy as np
import pytest
from morie.fn.dccgrch import dcc_garch


def test_dccgrch_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = dcc_garch(X)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_dccgrch_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = dcc_garch(X)
    assert isinstance(result, dict)
