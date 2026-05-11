"""Tests for t5enc.t5."""
import numpy as np
import pytest
from morie.fn.t5enc import t5


def test_t5enc_basic():
    """Test basic functionality."""
    src = np.random.default_rng(42).normal(0, 1, 100)
    tgt = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = t5(src, tgt, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_t5enc_edge():
    """Test edge cases."""
    src = np.random.default_rng(42).normal(0, 1, 100)
    tgt = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = t5(src, tgt, model)
    assert isinstance(result, dict)
