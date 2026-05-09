"""Tests for wilcox8u849.wilcox_chapter_8_unnumbered_849."""
import numpy as np
import pytest
from moirais.fn.wilcox8u849 import wilcox_chapter_8_unnumbered_849


def test_wilcox8u849_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_849(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox8u849_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_849(x)
    assert isinstance(result, dict)
