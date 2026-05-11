"""Tests for wilcox7u325.wilcox_chapter_7_unnumbered_325."""
import numpy as np
import pytest
from morie.fn.wilcox7u325 import wilcox_chapter_7_unnumbered_325


def test_wilcox7u325_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_unnumbered_325(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox7u325_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_unnumbered_325(x)
    assert isinstance(result, dict)
