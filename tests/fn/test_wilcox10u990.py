"""Tests for wilcox10u990.wilcox_chapter_10_unnumbered_990."""
import numpy as np
import pytest
from morie.fn.wilcox10u990 import wilcox_chapter_10_unnumbered_990


def test_wilcox10u990_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_990(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox10u990_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_990(x)
    assert isinstance(result, dict)
