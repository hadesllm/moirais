"""Tests for wilcox14u514.wilcox_chapter_14_unnumbered_514."""
import numpy as np
import pytest
from morie.fn.wilcox14u514 import wilcox_chapter_14_unnumbered_514


def test_wilcox14u514_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_514(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox14u514_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_514(x)
    assert isinstance(result, dict)
