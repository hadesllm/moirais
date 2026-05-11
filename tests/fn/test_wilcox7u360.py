"""Tests for wilcox7u360.wilcox_chapter_7_unnumbered_360."""
import numpy as np
import pytest
from morie.fn.wilcox7u360 import wilcox_chapter_7_unnumbered_360


def test_wilcox7u360_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_unnumbered_360(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox7u360_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_unnumbered_360(x)
    assert isinstance(result, dict)
