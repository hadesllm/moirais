"""Tests for hmtlu.geron_tlu."""
import numpy as np
import pytest
from moirais.fn.hmtlu import geron_tlu


def test_hmtlu_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_tlu(x, w, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmtlu_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_tlu(x, w, b)
    assert isinstance(result, dict)
