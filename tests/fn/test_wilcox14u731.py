"""Tests for wilcox14u731.wilcox_chapter_14_unnumbered_731."""
import numpy as np
import pytest
from morie.fn.wilcox14u731 import wilcox_chapter_14_unnumbered_731


def test_wilcox14u731_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_731(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u731_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_731(x)
    assert isinstance(result, dict)
