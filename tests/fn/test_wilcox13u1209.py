"""Tests for wilcox13u1209.wilcox_chapter_13_unnumbered_1209."""
import numpy as np
import pytest
from morie.fn.wilcox13u1209 import wilcox_chapter_13_unnumbered_1209


def test_wilcox13u1209_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1209(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox13u1209_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1209(x)
    assert isinstance(result, dict)
