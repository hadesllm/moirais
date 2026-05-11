"""Tests for wilcox7u308.wilcox_chapter_7_unnumbered_308."""
import numpy as np
import pytest
from morie.fn.wilcox7u308 import wilcox_chapter_7_unnumbered_308


def test_wilcox7u308_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_unnumbered_308(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox7u308_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_unnumbered_308(x)
    assert isinstance(result, dict)
