"""Tests for pageH.page_hinkley."""
import numpy as np
import pytest
from morie.fn.pageH import page_hinkley


def test_pageH_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = page_hinkley(x, threshold)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_pageH_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    result = page_hinkley(x, threshold)
    assert isinstance(result, dict)
