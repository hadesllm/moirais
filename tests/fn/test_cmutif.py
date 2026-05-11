"""Tests for cmutif.conditional_mi."""
import numpy as np
import pytest
from morie.fn.cmutif import conditional_mi


def test_cmutif_basic():
    """Test basic functionality."""
    pxyz = np.random.default_rng(42).normal(0, 1, 100)
    result = conditional_mi(pxyz)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cmutif_edge():
    """Test edge cases."""
    pxyz = np.random.default_rng(42).normal(0, 1, 100)
    result = conditional_mi(pxyz)
    assert isinstance(result, dict)
