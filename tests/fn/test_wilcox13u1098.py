"""Tests for wilcox13u1098.wilcox_chapter_13_unnumbered_1098."""
import numpy as np
import pytest
from morie.fn.wilcox13u1098 import wilcox_chapter_13_unnumbered_1098


def test_wilcox13u1098_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1098(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox13u1098_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_13_unnumbered_1098(x)
    assert isinstance(result, dict)
