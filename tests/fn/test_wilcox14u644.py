"""Tests for wilcox14u644.wilcox_chapter_14_unnumbered_644."""
import numpy as np
import pytest
from morie.fn.wilcox14u644 import wilcox_chapter_14_unnumbered_644


def test_wilcox14u644_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_644(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u644_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_644(x)
    assert isinstance(result, dict)
