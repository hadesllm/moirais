"""Tests for wilcox4u148.wilcox_chapter_4_unnumbered_148."""
import numpy as np
import pytest
from morie.fn.wilcox4u148 import wilcox_chapter_4_unnumbered_148


def test_wilcox4u148_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_148(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u148_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_148(x)
    assert isinstance(result, dict)
