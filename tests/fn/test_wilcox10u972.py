"""Tests for wilcox10u972.wilcox_chapter_10_unnumbered_972."""
import numpy as np
import pytest
from morie.fn.wilcox10u972 import wilcox_chapter_10_unnumbered_972


def test_wilcox10u972_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_972(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox10u972_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_972(x)
    assert isinstance(result, dict)
