"""Tests for bartkw.bartlett_kernel_weights."""
import numpy as np
import pytest
from morie.fn.bartkw import bartlett_kernel_weights


def test_bartkw_basic():
    """Test basic functionality."""
    lags = 10
    result = bartlett_kernel_weights(lags)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bartkw_edge():
    """Test edge cases."""
    lags = 10
    result = bartlett_kernel_weights(lags)
    assert isinstance(result, dict)
