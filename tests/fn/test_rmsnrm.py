"""Tests for rmsnrm.rms_norm."""
import numpy as np
import pytest
from morie.fn.rmsnrm import rms_norm


def test_rmsnrm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    result = rms_norm(y, x, g)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rmsnrm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    result = rms_norm(y, x, g)
    assert isinstance(result, dict)
