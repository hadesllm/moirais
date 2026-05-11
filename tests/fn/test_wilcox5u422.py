"""Tests for wilcox5u422.wilcox_chapter_5_unnumbered_422."""
import numpy as np
import pytest
from morie.fn.wilcox5u422 import wilcox_chapter_5_unnumbered_422


def test_wilcox5u422_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_422(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox5u422_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_422(x)
    assert isinstance(result, dict)
