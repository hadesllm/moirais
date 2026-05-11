"""Tests for wilcox4u63.wilcox_chapter_4_unnumbered_63."""
import numpy as np
import pytest
from morie.fn.wilcox4u63 import wilcox_chapter_4_unnumbered_63


def test_wilcox4u63_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_63(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u63_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_63(x)
    assert isinstance(result, dict)
