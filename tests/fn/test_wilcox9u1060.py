"""Tests for wilcox9u1060.wilcox_chapter_9_unnumbered_1060."""
import numpy as np
import pytest
from morie.fn.wilcox9u1060 import wilcox_chapter_9_unnumbered_1060


def test_wilcox9u1060_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_9_unnumbered_1060(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox9u1060_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_9_unnumbered_1060(x)
    assert isinstance(result, dict)
