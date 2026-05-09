"""Tests for hmprcv.geron_perceiver."""
import numpy as np
import pytest
from moirais.fn.hmprcv import geron_perceiver


def test_hmprcv_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    latents = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = geron_perceiver(x, latents, n_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmprcv_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    latents = np.random.default_rng(42).normal(0, 1, 100)
    n_iter = 50
    result = geron_perceiver(x, latents, n_iter)
    assert isinstance(result, dict)
