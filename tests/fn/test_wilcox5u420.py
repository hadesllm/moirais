"""Tests for wilcox5u420.wilcox_chapter_5_unnumbered_420."""
import numpy as np
import pytest
from morie.fn.wilcox5u420 import wilcox_chapter_5_unnumbered_420


def test_wilcox5u420_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_420(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox5u420_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_420(x)
    assert isinstance(result, dict)
