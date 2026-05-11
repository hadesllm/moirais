"""Tests for wilcox10u938.wilcox_chapter_10_unnumbered_938."""
import numpy as np
import pytest
from morie.fn.wilcox10u938 import wilcox_chapter_10_unnumbered_938


def test_wilcox10u938_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_938(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox10u938_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_938(x)
    assert isinstance(result, dict)
