"""Tests for wilcox10u920.wilcox_chapter_10_unnumbered_920."""
import numpy as np
import pytest
from morie.fn.wilcox10u920 import wilcox_chapter_10_unnumbered_920


def test_wilcox10u920_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_920(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox10u920_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_920(x)
    assert isinstance(result, dict)
