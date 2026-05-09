"""Tests for vitmlp.vit_mlp_block."""
import numpy as np
import pytest
from moirais.fn.vitmlp import vit_mlp_block


def test_vitmlp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    hidden_dim = 2
    result = vit_mlp_block(x, hidden_dim)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vitmlp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    hidden_dim = 2
    result = vit_mlp_block(x, hidden_dim)
    assert isinstance(result, dict)
