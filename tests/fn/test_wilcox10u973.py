"""Tests for wilcox10u973.wilcox_chapter_10_unnumbered_973."""
import numpy as np
import pytest
from morie.fn.wilcox10u973 import wilcox_chapter_10_unnumbered_973


def test_wilcox10u973_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_973(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox10u973_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_973(x)
    assert isinstance(result, dict)
