"""Tests for kmprf.kamath_prefix_lm_mask."""
import numpy as np
import pytest
from morie.fn.kmprf import kamath_prefix_lm_mask


def test_kmprf_basic():
    """Test basic functionality."""
    prefix_len = np.random.default_rng(42).normal(0, 1, 100)
    total_len = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_prefix_lm_mask(prefix_len, total_len)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmprf_edge():
    """Test edge cases."""
    prefix_len = np.random.default_rng(42).normal(0, 1, 100)
    total_len = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_prefix_lm_mask(prefix_len, total_len)
    assert isinstance(result, dict)
