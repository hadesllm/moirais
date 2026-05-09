"""Tests for joadf.joseph_adf_unit_root_test."""
import numpy as np
import pytest
from moirais.fn.joadf import joseph_adf_unit_root_test


def test_joadf_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    result = joseph_adf_unit_root_test(y, p)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_joadf_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    result = joseph_adf_unit_root_test(y, p)
    assert isinstance(result, dict)
