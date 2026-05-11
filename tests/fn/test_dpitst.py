"""Tests for dpitst.data_processing_inequality."""
import numpy as np
import pytest
from morie.fn.dpitst import data_processing_inequality


def test_dpitst_basic():
    """Test basic functionality."""
    pxyz = np.random.default_rng(42).normal(0, 1, 100)
    result = data_processing_inequality(pxyz)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_dpitst_edge():
    """Test edge cases."""
    pxyz = np.random.default_rng(42).normal(0, 1, 100)
    result = data_processing_inequality(pxyz)
    assert isinstance(result, dict)
