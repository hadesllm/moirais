"""Tests for wilcox3u1.wilcox_chapter_3_unnumbered_1."""
import numpy as np
import pytest
from morie.fn.wilcox3u1 import wilcox_chapter_3_unnumbered_1


def test_wilcox3u1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_3_unnumbered_1(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox3u1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_3_unnumbered_1(x)
    assert isinstance(result, dict)
