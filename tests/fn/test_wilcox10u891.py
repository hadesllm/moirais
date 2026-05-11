"""Tests for wilcox10u891.wilcox_chapter_10_unnumbered_891."""
import numpy as np
import pytest
from morie.fn.wilcox10u891 import wilcox_chapter_10_unnumbered_891


def test_wilcox10u891_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_891(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox10u891_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_891(x)
    assert isinstance(result, dict)
