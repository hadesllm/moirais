"""Tests for chgseg.changepoint_segmentation."""
import numpy as np
import pytest
from morie.fn.chgseg import changepoint_segmentation


def test_chgseg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    penalty = np.random.default_rng(42).normal(0, 1, 100)
    result = changepoint_segmentation(y, penalty)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_chgseg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    penalty = np.random.default_rng(42).normal(0, 1, 100)
    result = changepoint_segmentation(y, penalty)
    assert isinstance(result, dict)
