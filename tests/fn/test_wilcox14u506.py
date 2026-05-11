"""Tests for wilcox14u506.wilcox_chapter_14_unnumbered_506."""
import numpy as np
import pytest
from morie.fn.wilcox14u506 import wilcox_chapter_14_unnumbered_506


def test_wilcox14u506_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_506(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_wilcox14u506_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_506(x)
    assert isinstance(result, dict)
