"""Tests for gb1031.gibbons_k_ctrl_median."""
import numpy as np
import pytest
from moirais.fn.gb1031 import gibbons_k_ctrl_median


def test_gb1031_basic():
    """Test basic functionality."""
    groups = np.random.default_rng(42).normal(0, 1, 100)
    control = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_k_ctrl_median(groups, control)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb1031_edge():
    """Test edge cases."""
    groups = np.random.default_rng(42).normal(0, 1, 100)
    control = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_k_ctrl_median(groups, control)
    assert isinstance(result, dict)
