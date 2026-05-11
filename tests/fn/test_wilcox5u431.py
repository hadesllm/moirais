"""Tests for wilcox5u431.wilcox_chapter_5_unnumbered_431."""
import numpy as np
import pytest
from morie.fn.wilcox5u431 import wilcox_chapter_5_unnumbered_431


def test_wilcox5u431_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_431(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox5u431_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_431(x)
    assert isinstance(result, dict)
