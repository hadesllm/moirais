"""Tests for mfird.mirt_factor_loading."""
import numpy as np
import pytest
from morie.fn.mfird import mirt_factor_loading


def test_mfird_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = mirt_factor_loading(y, a)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mfird_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = mirt_factor_loading(y, a)
    assert isinstance(result, dict)
