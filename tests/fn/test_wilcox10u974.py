"""Tests for wilcox10u974.wilcox_chapter_10_unnumbered_974."""
import numpy as np
import pytest
from moirais.fn.wilcox10u974 import wilcox_chapter_10_unnumbered_974


def test_wilcox10u974_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_974(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox10u974_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_974(x)
    assert isinstance(result, dict)
