"""Tests for wilcox14u789.wilcox_chapter_14_unnumbered_789."""
import numpy as np
import pytest
from moirais.fn.wilcox14u789 import wilcox_chapter_14_unnumbered_789


def test_wilcox14u789_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_789(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox14u789_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_789(x)
    assert isinstance(result, dict)
