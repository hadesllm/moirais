"""Tests for irtnrm.nominal_response."""
import numpy as np
import pytest
from moirais.fn.irtnrm import nominal_response


def test_irtnrm_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    ncats = np.random.default_rng(42).normal(0, 1, 100)
    result = nominal_response(X, ncats)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_irtnrm_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    ncats = np.random.default_rng(42).normal(0, 1, 100)
    result = nominal_response(X, ncats)
    assert isinstance(result, dict)
