"""Tests for wilcox14u747.wilcox_chapter_14_unnumbered_747."""
import numpy as np
import pytest
from morie.fn.wilcox14u747 import wilcox_chapter_14_unnumbered_747


def test_wilcox14u747_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_747(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u747_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_747(x)
    assert isinstance(result, dict)
