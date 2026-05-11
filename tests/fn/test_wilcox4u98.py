"""Tests for wilcox4u98.wilcox_chapter_4_unnumbered_98."""
import numpy as np
import pytest
from morie.fn.wilcox4u98 import wilcox_chapter_4_unnumbered_98


def test_wilcox4u98_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_98(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u98_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_98(x)
    assert isinstance(result, dict)
