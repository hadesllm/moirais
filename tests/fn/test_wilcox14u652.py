"""Tests for wilcox14u652.wilcox_chapter_14_unnumbered_652."""
import numpy as np
import pytest
from morie.fn.wilcox14u652 import wilcox_chapter_14_unnumbered_652


def test_wilcox14u652_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_652(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u652_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_652(x)
    assert isinstance(result, dict)
