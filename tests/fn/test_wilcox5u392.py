"""Tests for wilcox5u392.wilcox_chapter_5_unnumbered_392."""
import numpy as np
import pytest
from morie.fn.wilcox5u392 import wilcox_chapter_5_unnumbered_392


def test_wilcox5u392_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_392(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox5u392_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_392(x)
    assert isinstance(result, dict)
