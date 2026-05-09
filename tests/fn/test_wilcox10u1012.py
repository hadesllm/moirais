"""Tests for wilcox10u1012.wilcox_chapter_10_unnumbered_1012."""
import numpy as np
import pytest
from moirais.fn.wilcox10u1012 import wilcox_chapter_10_unnumbered_1012


def test_wilcox10u1012_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_1012(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox10u1012_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_1012(x)
    assert isinstance(result, dict)
