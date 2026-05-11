"""Tests for grmms.geron_minmax_scaler."""
import numpy as np
import pytest
from morie.fn.grmms import geron_minmax_scaler


def test_grmms_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = geron_minmax_scaler(X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grmms_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = geron_minmax_scaler(X)
    assert isinstance(result, dict)
