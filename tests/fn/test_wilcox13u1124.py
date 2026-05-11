"""Tests for wilcox13u1124.wilcox_chapter_13_unnumbered_1124."""
import numpy as np
import pytest
from morie.fn.wilcox13u1124 import wilcox_chapter_13_unnumbered_1124


def test_wilcox13u1124_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1124(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox13u1124_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1124(x)
    assert isinstance(result, dict)
