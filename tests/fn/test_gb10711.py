"""Tests for gb10711.gibbons_ctrl_normal_asymp."""
import numpy as np
import pytest
from morie.fn.gb10711 import gibbons_ctrl_normal_asymp


def test_gb10711_basic():
    """Test basic functionality."""
    groups = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_ctrl_normal_asymp(groups)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb10711_edge():
    """Test edge cases."""
    groups = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_ctrl_normal_asymp(groups)
    assert isinstance(result, dict)
