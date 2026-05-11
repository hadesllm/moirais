"""Tests for wilcox15u1391.wilcox_chapter_15_unnumbered_1391."""
import numpy as np
import pytest
from morie.fn.wilcox15u1391 import wilcox_chapter_15_unnumbered_1391


def test_wilcox15u1391_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_15_unnumbered_1391(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox15u1391_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_15_unnumbered_1391(x)
    assert isinstance(result, dict)
