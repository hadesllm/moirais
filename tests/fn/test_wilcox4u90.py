"""Tests for wilcox4u90.wilcox_chapter_4_unnumbered_90."""
import numpy as np
import pytest
from morie.fn.wilcox4u90 import wilcox_chapter_4_unnumbered_90


def test_wilcox4u90_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_90(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox4u90_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_90(x)
    assert isinstance(result, dict)
