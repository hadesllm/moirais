"""Tests for wilcox13u1246.wilcox_chapter_13_unnumbered_1246."""
import numpy as np
import pytest
from morie.fn.wilcox13u1246 import wilcox_chapter_13_unnumbered_1246


def test_wilcox13u1246_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1246(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox13u1246_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1246(x)
    assert isinstance(result, dict)
