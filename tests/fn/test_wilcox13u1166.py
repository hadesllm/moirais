"""Tests for wilcox13u1166.wilcox_chapter_13_unnumbered_1166."""
import numpy as np
import pytest
from moirais.fn.wilcox13u1166 import wilcox_chapter_13_unnumbered_1166


def test_wilcox13u1166_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1166(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_wilcox13u1166_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1166(x)
    assert isinstance(result, dict)
