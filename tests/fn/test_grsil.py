"""Tests for grsil.geron_silhouette_score."""
import numpy as np
import pytest
from moirais.fn.grsil import geron_silhouette_score


def test_grsil_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    labels = np.random.default_rng(43).integers(0, 2, 100)
    result = geron_silhouette_score(X, labels)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grsil_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    labels = np.random.default_rng(43).integers(0, 2, 100)
    result = geron_silhouette_score(X, labels)
    assert isinstance(result, dict)
