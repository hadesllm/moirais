"""Tests for hmpe.geron_positional_encoding."""
import numpy as np
import pytest
from morie.fn.hmpe import geron_positional_encoding


def test_hmpe_basic():
    """Test basic functionality."""
    pos = np.random.default_rng(42).normal(0, 1, 100)
    d_model = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_positional_encoding(pos, d_model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmpe_edge():
    """Test edge cases."""
    pos = np.random.default_rng(42).normal(0, 1, 100)
    d_model = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_positional_encoding(pos, d_model)
    assert isinstance(result, dict)
