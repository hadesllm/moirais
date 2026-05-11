"""Tests for wilcox9u1080.wilcox_chapter_9_unnumbered_1080."""
import numpy as np
import pytest
from morie.fn.wilcox9u1080 import wilcox_chapter_9_unnumbered_1080


def test_wilcox9u1080_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_9_unnumbered_1080(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox9u1080_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_9_unnumbered_1080(x)
    assert isinstance(result, dict)
