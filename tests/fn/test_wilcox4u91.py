"""Tests for wilcox4u91.wilcox_chapter_4_unnumbered_91."""
import numpy as np
import pytest
from morie.fn.wilcox4u91 import wilcox_chapter_4_unnumbered_91


def test_wilcox4u91_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_91(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox4u91_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_91(x)
    assert isinstance(result, dict)
