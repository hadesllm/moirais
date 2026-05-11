"""Tests for vitfwd.vit_forward."""
import numpy as np
import pytest
from morie.fn.vitfwd import vit_forward


def test_vitfwd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    patch_size = 100
    embed_dim = 2
    num_heads = np.random.default_rng(42).normal(0, 1, 100)
    num_layers = np.random.default_rng(42).normal(0, 1, 100)
    result = vit_forward(x, patch_size, embed_dim, num_heads, num_layers)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vitfwd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    patch_size = 100
    embed_dim = 2
    num_heads = np.random.default_rng(42).normal(0, 1, 100)
    num_layers = np.random.default_rng(42).normal(0, 1, 100)
    result = vit_forward(x, patch_size, embed_dim, num_heads, num_layers)
    assert isinstance(result, dict)
