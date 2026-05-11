"""Tests for grpcap.geron_pca_projection."""
import numpy as np
import pytest
from morie.fn.grpcap import geron_pca_projection


def test_grpcap_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    d = 5
    result = geron_pca_projection(X, d)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grpcap_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    d = 5
    result = geron_pca_projection(X, d)
    assert isinstance(result, dict)
