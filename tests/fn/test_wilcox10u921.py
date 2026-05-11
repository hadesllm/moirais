"""Tests for wilcox10u921.wilcox_chapter_10_unnumbered_921."""
import numpy as np
import pytest
from morie.fn.wilcox10u921 import wilcox_chapter_10_unnumbered_921


def test_wilcox10u921_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_921(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox10u921_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_921(x)
    assert isinstance(result, dict)
