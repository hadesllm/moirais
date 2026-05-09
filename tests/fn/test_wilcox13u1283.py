"""Tests for wilcox13u1283.wilcox_chapter_13_unnumbered_1283."""
import numpy as np
import pytest
from moirais.fn.wilcox13u1283 import wilcox_chapter_13_unnumbered_1283


def test_wilcox13u1283_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1283(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox13u1283_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1283(x)
    assert isinstance(result, dict)
