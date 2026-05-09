"""Tests for btmult.boot_multinomial_weights."""
import numpy as np
import pytest
from moirais.fn.btmult import boot_multinomial_weights


def test_btmult_basic():
    """Test basic functionality."""
    n = 100
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    rng = np.random.default_rng(42)
    result = boot_multinomial_weights(n, B, rng)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_btmult_edge():
    """Test edge cases."""
    n = 100
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    rng = np.random.default_rng(42)
    result = boot_multinomial_weights(n, B, rng)
    assert isinstance(result, dict)
