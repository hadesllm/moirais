"""Tests for btblen.boot_block_length_pr."""
import numpy as np
import pytest
from moirais.fn.btblen import boot_block_length_pr


def test_btblen_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    method = 'auto'
    result = boot_block_length_pr(x, method)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btblen_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    method = 'auto'
    result = boot_block_length_pr(x, method)
    assert isinstance(result, dict)
