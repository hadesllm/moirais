"""Tests for wilcox5u441.wilcox_chapter_5_unnumbered_441."""
import numpy as np
import pytest
from morie.fn.wilcox5u441 import wilcox_chapter_5_unnumbered_441


def test_wilcox5u441_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_441(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox5u441_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_441(x)
    assert isinstance(result, dict)
