"""Tests for wilcox13u1301.wilcox_chapter_13_unnumbered_1301."""
import numpy as np
import pytest
from morie.fn.wilcox13u1301 import wilcox_chapter_13_unnumbered_1301


def test_wilcox13u1301_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1301(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox13u1301_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1301(x)
    assert isinstance(result, dict)
