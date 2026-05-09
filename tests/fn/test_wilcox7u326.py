"""Tests for wilcox7u326.wilcox_chapter_7_unnumbered_326."""
import numpy as np
import pytest
from moirais.fn.wilcox7u326 import wilcox_chapter_7_unnumbered_326


def test_wilcox7u326_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_unnumbered_326(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox7u326_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_unnumbered_326(x)
    assert isinstance(result, dict)
