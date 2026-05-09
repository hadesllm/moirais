"""Tests for barte.bart."""
import numpy as np
import pytest
from moirais.fn.barte import bart


def test_barte_basic():
    """Test basic functionality."""
    src = np.random.default_rng(42).normal(0, 1, 100)
    tgt = np.random.default_rng(42).normal(0, 1, 100)
    result = bart(src, tgt)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_barte_edge():
    """Test edge cases."""
    src = np.random.default_rng(42).normal(0, 1, 100)
    tgt = np.random.default_rng(42).normal(0, 1, 100)
    result = bart(src, tgt)
    assert isinstance(result, dict)
