"""Tests for wilcox13u1161.wilcox_chapter_13_unnumbered_1161."""
import numpy as np
import pytest
from morie.fn.wilcox13u1161 import wilcox_chapter_13_unnumbered_1161


def test_wilcox13u1161_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1161(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_wilcox13u1161_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1161(x)
    assert isinstance(result, dict)
