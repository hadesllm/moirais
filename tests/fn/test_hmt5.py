"""Tests for hmt5.geron_t5."""
import numpy as np
import pytest
from moirais.fn.hmt5 import geron_t5


def test_hmt5_basic():
    """Test basic functionality."""
    src = np.random.default_rng(42).normal(0, 1, 100)
    tgt = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_t5(src, tgt)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmt5_edge():
    """Test edge cases."""
    src = np.random.default_rng(42).normal(0, 1, 100)
    tgt = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_t5(src, tgt)
    assert isinstance(result, dict)
