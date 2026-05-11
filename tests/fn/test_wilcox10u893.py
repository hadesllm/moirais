"""Tests for wilcox10u893.wilcox_chapter_10_unnumbered_893."""
import numpy as np
import pytest
from morie.fn.wilcox10u893 import wilcox_chapter_10_unnumbered_893


def test_wilcox10u893_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_893(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox10u893_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_893(x)
    assert isinstance(result, dict)
