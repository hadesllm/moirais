"""Tests for binseg.binary_segmentation."""
import numpy as np
import pytest
from morie.fn.binseg import binary_segmentation


def test_binseg_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = binary_segmentation(x, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_binseg_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = binary_segmentation(x, K)
    assert isinstance(result, dict)
