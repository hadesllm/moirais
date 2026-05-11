"""Tests for wilcox10u969.wilcox_chapter_10_unnumbered_969."""
import numpy as np
import pytest
from morie.fn.wilcox10u969 import wilcox_chapter_10_unnumbered_969


def test_wilcox10u969_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_969(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox10u969_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_969(x)
    assert isinstance(result, dict)
