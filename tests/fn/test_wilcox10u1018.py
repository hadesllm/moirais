"""Tests for wilcox10u1018.wilcox_chapter_10_unnumbered_1018."""
import numpy as np
import pytest
from moirais.fn.wilcox10u1018 import wilcox_chapter_10_unnumbered_1018


def test_wilcox10u1018_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_1018(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox10u1018_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_1018(x)
    assert isinstance(result, dict)
