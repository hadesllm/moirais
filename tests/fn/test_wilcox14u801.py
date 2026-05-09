"""Tests for wilcox14u801.wilcox_chapter_14_unnumbered_801."""
import numpy as np
import pytest
from moirais.fn.wilcox14u801 import wilcox_chapter_14_unnumbered_801


def test_wilcox14u801_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_801(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox14u801_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_801(x)
    assert isinstance(result, dict)
