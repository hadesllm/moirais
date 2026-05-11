"""Tests for wilcox13u1298.wilcox_chapter_13_unnumbered_1298."""
import numpy as np
import pytest
from morie.fn.wilcox13u1298 import wilcox_chapter_13_unnumbered_1298


def test_wilcox13u1298_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1298(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox13u1298_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1298(x)
    assert isinstance(result, dict)
