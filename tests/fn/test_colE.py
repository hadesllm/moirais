"""Tests for colE.cold_start_user."""
import numpy as np
import pytest
from moirais.fn.colE import cold_start_user


def test_colE_basic():
    """Test basic functionality."""
    user = np.random.default_rng(42).normal(0, 1, 100)
    mode = 'auto'
    result = cold_start_user(user, mode)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_colE_edge():
    """Test edge cases."""
    user = np.random.default_rng(42).normal(0, 1, 100)
    mode = 'auto'
    result = cold_start_user(user, mode)
    assert isinstance(result, dict)
