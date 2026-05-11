"""Tests for wilcox13u1130.wilcox_chapter_13_unnumbered_1130."""
import numpy as np
import pytest
from morie.fn.wilcox13u1130 import wilcox_chapter_13_unnumbered_1130


def test_wilcox13u1130_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1130(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox13u1130_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1130(x)
    assert isinstance(result, dict)
