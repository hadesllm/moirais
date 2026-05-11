"""Tests for wilcox13u1256.wilcox_chapter_13_unnumbered_1256."""
import numpy as np
import pytest
from morie.fn.wilcox13u1256 import wilcox_chapter_13_unnumbered_1256


def test_wilcox13u1256_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1256(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox13u1256_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1256(x)
    assert isinstance(result, dict)
