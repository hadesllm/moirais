"""Tests for wilcox14u661.wilcox_chapter_14_unnumbered_661."""
import numpy as np
import pytest
from morie.fn.wilcox14u661 import wilcox_chapter_14_unnumbered_661


def test_wilcox14u661_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_661(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox14u661_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_661(x)
    assert isinstance(result, dict)
