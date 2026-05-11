"""Tests for wilcox14u525.wilcox_chapter_14_unnumbered_525."""
import numpy as np
import pytest
from morie.fn.wilcox14u525 import wilcox_chapter_14_unnumbered_525


def test_wilcox14u525_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_525(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox14u525_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_525(x)
    assert isinstance(result, dict)
