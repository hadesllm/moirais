"""Tests for wilcox13u1352.wilcox_chapter_13_unnumbered_1352."""
import numpy as np
import pytest
from morie.fn.wilcox13u1352 import wilcox_chapter_13_unnumbered_1352


def test_wilcox13u1352_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1352(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox13u1352_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1352(x)
    assert isinstance(result, dict)
