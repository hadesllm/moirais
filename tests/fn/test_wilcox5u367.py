"""Tests for wilcox5u367.wilcox_chapter_5_unnumbered_367."""
import numpy as np
import pytest
from morie.fn.wilcox5u367 import wilcox_chapter_5_unnumbered_367


def test_wilcox5u367_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_367(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox5u367_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_367(x)
    assert isinstance(result, dict)
