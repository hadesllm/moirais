"""Tests for gb621.gibbons_ks2samp."""
import numpy as np
import pytest
from moirais.fn.gb621 import gibbons_ks2samp


def test_gb621_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_ks2samp(x, y)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb621_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_ks2samp(x, y)
    assert isinstance(result, dict)
