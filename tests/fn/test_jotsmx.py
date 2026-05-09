"""Tests for jotsmx.joseph_tsmixer."""
import numpy as np
import pytest
from moirais.fn.jotsmx import joseph_tsmixer


def test_jotsmx_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n_blocks = np.random.default_rng(42).normal(0, 1, 100)
    hidden_dim = 2
    result = joseph_tsmixer(x, n_blocks, hidden_dim)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jotsmx_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n_blocks = np.random.default_rng(42).normal(0, 1, 100)
    hidden_dim = 2
    result = joseph_tsmixer(x, n_blocks, hidden_dim)
    assert isinstance(result, dict)
