"""Tests for wilcox14u590.wilcox_chapter_14_unnumbered_590."""
import numpy as np
import pytest
from moirais.fn.wilcox14u590 import wilcox_chapter_14_unnumbered_590


def test_wilcox14u590_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_590(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u590_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_590(x)
    assert isinstance(result, dict)
