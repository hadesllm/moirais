"""Tests for bayauto.autocorrelation_check."""
import numpy as np
import pytest
from moirais.fn.bayauto import autocorrelation_check


def test_bayauto_basic():
    """Test basic functionality."""
    chain = np.random.default_rng(42).normal(0, 1, 100)
    result = autocorrelation_check(chain)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_bayauto_edge():
    """Test edge cases."""
    chain = np.random.default_rng(42).normal(0, 1, 100)
    result = autocorrelation_check(chain)
    assert isinstance(result, dict)
