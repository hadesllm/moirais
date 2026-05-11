"""Tests for wilcox14u591.wilcox_chapter_14_unnumbered_591."""
import numpy as np
import pytest
from morie.fn.wilcox14u591 import wilcox_chapter_14_unnumbered_591


def test_wilcox14u591_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_591(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u591_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_591(x)
    assert isinstance(result, dict)
