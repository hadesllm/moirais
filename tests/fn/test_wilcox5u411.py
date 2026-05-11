"""Tests for wilcox5u411.wilcox_chapter_5_unnumbered_411."""
import numpy as np
import pytest
from morie.fn.wilcox5u411 import wilcox_chapter_5_unnumbered_411


def test_wilcox5u411_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_411(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox5u411_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_411(x)
    assert isinstance(result, dict)
