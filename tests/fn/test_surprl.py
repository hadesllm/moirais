"""Tests for surprl.surprisal."""
import numpy as np
import pytest
from moirais.fn.surprl import surprisal


def test_surprl_basic():
    """Test basic functionality."""
    p = 5
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = surprisal(p, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_surprl_edge():
    """Test edge cases."""
    p = 5
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = surprisal(p, x)
    assert isinstance(result, dict)
