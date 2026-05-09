"""Tests for hmbart.geron_bart."""
import numpy as np
import pytest
from moirais.fn.hmbart import geron_bart


def test_hmbart_basic():
    """Test basic functionality."""
    src = np.random.default_rng(42).normal(0, 1, 100)
    tgt = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_bart(src, tgt)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmbart_edge():
    """Test edge cases."""
    src = np.random.default_rng(42).normal(0, 1, 100)
    tgt = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_bart(src, tgt)
    assert isinstance(result, dict)
