"""Tests for hmpd.geron_padding."""
import numpy as np
import pytest
from moirais.fn.hmpd import geron_padding


def test_hmpd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    pad_h = np.random.default_rng(42).normal(0, 1, 100)
    pad_w = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_padding(x, pad_h, pad_w)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmpd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    pad_h = np.random.default_rng(42).normal(0, 1, 100)
    pad_w = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_padding(x, pad_h, pad_w)
    assert isinstance(result, dict)
