"""Tests for wilcox7u333.wilcox_chapter_7_unnumbered_333."""
import numpy as np
import pytest
from morie.fn.wilcox7u333 import wilcox_chapter_7_unnumbered_333


def test_wilcox7u333_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_unnumbered_333(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox7u333_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_unnumbered_333(x)
    assert isinstance(result, dict)
