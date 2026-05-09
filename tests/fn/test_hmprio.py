"""Tests for hmprio.geron_perceiver_io."""
import numpy as np
import pytest
from moirais.fn.hmprio import geron_perceiver_io


def test_hmprio_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    latents = np.random.default_rng(42).normal(0, 1, 100)
    queries = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_perceiver_io(x, latents, queries)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmprio_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    latents = np.random.default_rng(42).normal(0, 1, 100)
    queries = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_perceiver_io(x, latents, queries)
    assert isinstance(result, dict)
