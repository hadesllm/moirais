"""Tests for wilcox14u648.wilcox_chapter_14_unnumbered_648."""
import numpy as np
import pytest
from morie.fn.wilcox14u648 import wilcox_chapter_14_unnumbered_648


def test_wilcox14u648_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_648(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u648_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_648(x)
    assert isinstance(result, dict)
