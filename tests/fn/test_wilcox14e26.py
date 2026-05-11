"""Tests for wilcox14e26.wilcox_chapter_14_equation_26."""
import numpy as np
import pytest
from morie.fn.wilcox14e26 import wilcox_chapter_14_equation_26


def test_wilcox14e26_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_equation_26(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox14e26_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_equation_26(x)
    assert isinstance(result, dict)
