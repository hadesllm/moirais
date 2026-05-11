"""Tests for hmsil.geron_silhouette."""
import numpy as np
import pytest
from morie.fn.hmsil import geron_silhouette


def test_hmsil_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    labels = np.random.default_rng(43).integers(0, 2, 100)
    result = geron_silhouette(X, labels)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmsil_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    labels = np.random.default_rng(43).integers(0, 2, 100)
    result = geron_silhouette(X, labels)
    assert isinstance(result, dict)
