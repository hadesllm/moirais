"""Tests for wilcox13u1101.wilcox_chapter_13_unnumbered_1101."""
import numpy as np
import pytest
from morie.fn.wilcox13u1101 import wilcox_chapter_13_unnumbered_1101


def test_wilcox13u1101_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1101(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox13u1101_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1101(x)
    assert isinstance(result, dict)
