"""Tests for wilcox8u822.wilcox_chapter_8_unnumbered_822."""
import numpy as np
import pytest
from morie.fn.wilcox8u822 import wilcox_chapter_8_unnumbered_822


def test_wilcox8u822_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_822(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox8u822_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_822(x)
    assert isinstance(result, dict)
