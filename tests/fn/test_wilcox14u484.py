"""Tests for wilcox14u484.wilcox_chapter_14_unnumbered_484."""
import numpy as np
import pytest
from morie.fn.wilcox14u484 import wilcox_chapter_14_unnumbered_484


def test_wilcox14u484_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_484(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u484_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_484(x)
    assert isinstance(result, dict)
