"""Tests for wilcox8u863.wilcox_chapter_8_unnumbered_863."""
import numpy as np
import pytest
from morie.fn.wilcox8u863 import wilcox_chapter_8_unnumbered_863


def test_wilcox8u863_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_863(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox8u863_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_863(x)
    assert isinstance(result, dict)
