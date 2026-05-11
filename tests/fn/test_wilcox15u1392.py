"""Tests for wilcox15u1392.wilcox_chapter_15_unnumbered_1392."""
import numpy as np
import pytest
from morie.fn.wilcox15u1392 import wilcox_chapter_15_unnumbered_1392


def test_wilcox15u1392_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_15_unnumbered_1392(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox15u1392_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_15_unnumbered_1392(x)
    assert isinstance(result, dict)
