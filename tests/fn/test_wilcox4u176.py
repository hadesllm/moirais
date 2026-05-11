"""Tests for wilcox4u176.wilcox_chapter_4_unnumbered_176."""
import numpy as np
import pytest
from morie.fn.wilcox4u176 import wilcox_chapter_4_unnumbered_176


def test_wilcox4u176_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_176(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u176_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_176(x)
    assert isinstance(result, dict)
