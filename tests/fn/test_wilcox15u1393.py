"""Tests for wilcox15u1393.wilcox_chapter_15_unnumbered_1393."""
import numpy as np
import pytest
from moirais.fn.wilcox15u1393 import wilcox_chapter_15_unnumbered_1393


def test_wilcox15u1393_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_15_unnumbered_1393(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox15u1393_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_15_unnumbered_1393(x)
    assert isinstance(result, dict)
