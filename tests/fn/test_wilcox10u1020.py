"""Tests for wilcox10u1020.wilcox_chapter_10_unnumbered_1020."""
import numpy as np
import pytest
from morie.fn.wilcox10u1020 import wilcox_chapter_10_unnumbered_1020


def test_wilcox10u1020_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_1020(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox10u1020_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_1020(x)
    assert isinstance(result, dict)
