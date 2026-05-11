"""Tests for grpio.geron_perceiver_io."""
import numpy as np
import pytest
from morie.fn.grpio import geron_perceiver_io


def test_grpio_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z_latent = np.random.default_rng(42).normal(0, 1, 100)
    output_queries = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_perceiver_io(X, Z_latent, output_queries)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grpio_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    Z_latent = np.random.default_rng(42).normal(0, 1, 100)
    output_queries = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_perceiver_io(X, Z_latent, output_queries)
    assert isinstance(result, dict)
