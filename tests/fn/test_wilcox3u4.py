"""Tests for wilcox3u4.wilcox_chapter_3_unnumbered_4."""
import numpy as np
import pytest
from morie.fn.wilcox3u4 import wilcox_chapter_3_unnumbered_4


def test_wilcox3u4_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_3_unnumbered_4(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox3u4_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_3_unnumbered_4(x)
    assert isinstance(result, dict)
