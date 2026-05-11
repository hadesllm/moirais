"""Tests for wilcox14u693.wilcox_chapter_14_unnumbered_693."""
import numpy as np
import pytest
from morie.fn.wilcox14u693 import wilcox_chapter_14_unnumbered_693


def test_wilcox14u693_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_693(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u693_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_693(x)
    assert isinstance(result, dict)
