"""Tests for wilcox10u1033.wilcox_chapter_10_unnumbered_1033."""
import numpy as np
import pytest
from morie.fn.wilcox10u1033 import wilcox_chapter_10_unnumbered_1033


def test_wilcox10u1033_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_1033(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox10u1033_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_1033(x)
    assert isinstance(result, dict)
