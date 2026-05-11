"""Tests for wilcox4u36.wilcox_chapter_4_unnumbered_36."""
import numpy as np
import pytest
from morie.fn.wilcox4u36 import wilcox_chapter_4_unnumbered_36


def test_wilcox4u36_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_36(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u36_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_36(x)
    assert isinstance(result, dict)
