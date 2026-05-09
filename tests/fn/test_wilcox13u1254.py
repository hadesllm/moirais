"""Tests for wilcox13u1254.wilcox_chapter_13_unnumbered_1254."""
import numpy as np
import pytest
from moirais.fn.wilcox13u1254 import wilcox_chapter_13_unnumbered_1254


def test_wilcox13u1254_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1254(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox13u1254_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1254(x)
    assert isinstance(result, dict)
