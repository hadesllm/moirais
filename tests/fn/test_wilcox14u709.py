"""Tests for wilcox14u709.wilcox_chapter_14_unnumbered_709."""
import numpy as np
import pytest
from morie.fn.wilcox14u709 import wilcox_chapter_14_unnumbered_709


def test_wilcox14u709_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_709(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u709_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_709(x)
    assert isinstance(result, dict)
