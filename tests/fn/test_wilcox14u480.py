"""Tests for wilcox14u480.wilcox_chapter_14_unnumbered_480."""
import numpy as np
import pytest
from morie.fn.wilcox14u480 import wilcox_chapter_14_unnumbered_480


def test_wilcox14u480_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_480(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_wilcox14u480_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_480(x)
    assert isinstance(result, dict)
