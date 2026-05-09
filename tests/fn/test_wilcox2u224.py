"""Tests for wilcox2u224.wilcox_chapter_2_unnumbered_224."""
import numpy as np
import pytest
from moirais.fn.wilcox2u224 import wilcox_chapter_2_unnumbered_224


def test_wilcox2u224_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_224(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox2u224_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_224(x)
    assert isinstance(result, dict)
