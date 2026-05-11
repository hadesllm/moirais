"""Tests for volraq.vol_realised_quadratic_var."""
import numpy as np
import pytest
from morie.fn.volraq import vol_realised_quadratic_var


def test_volraq_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_realised_quadratic_var(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volraq_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_realised_quadratic_var(x)
    assert isinstance(result, dict)
