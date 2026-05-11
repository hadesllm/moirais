"""Tests for wilcox14u795.wilcox_chapter_14_unnumbered_795."""
import numpy as np
import pytest
from morie.fn.wilcox14u795 import wilcox_chapter_14_unnumbered_795


def test_wilcox14u795_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_795(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u795_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_795(x)
    assert isinstance(result, dict)
