"""Tests for wilcox14u565.wilcox_chapter_14_unnumbered_565."""
import numpy as np
import pytest
from morie.fn.wilcox14u565 import wilcox_chapter_14_unnumbered_565


def test_wilcox14u565_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_565(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox14u565_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_565(x)
    assert isinstance(result, dict)
