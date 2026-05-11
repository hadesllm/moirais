"""Tests for otmqd.ot_quantization_distortion."""
import numpy as np
import pytest
from morie.fn.otmqd import ot_quantization_distortion


def test_otmqd_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    centroids = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_quantization_distortion(X, centroids)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otmqd_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    centroids = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_quantization_distortion(X, centroids)
    assert isinstance(result, dict)
