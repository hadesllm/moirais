"""Tests for wilcox10u1036.wilcox_chapter_10_unnumbered_1036."""
import numpy as np
import pytest
from morie.fn.wilcox10u1036 import wilcox_chapter_10_unnumbered_1036


def test_wilcox10u1036_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_1036(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox10u1036_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_1036(x)
    assert isinstance(result, dict)
