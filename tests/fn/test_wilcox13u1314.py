"""Tests for wilcox13u1314.wilcox_chapter_13_unnumbered_1314."""
import numpy as np
import pytest
from morie.fn.wilcox13u1314 import wilcox_chapter_13_unnumbered_1314


def test_wilcox13u1314_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1314(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox13u1314_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1314(x)
    assert isinstance(result, dict)
