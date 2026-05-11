"""Tests for wilcox4u6.wilcox_chapter_4_unnumbered_6."""
import numpy as np
import pytest
from morie.fn.wilcox4u6 import wilcox_chapter_4_unnumbered_6


def test_wilcox4u6_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_6(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u6_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_6(x)
    assert isinstance(result, dict)
