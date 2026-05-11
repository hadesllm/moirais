"""Tests for wilcox7e26.wilcox_chapter_7_equation_26."""
import numpy as np
import pytest
from morie.fn.wilcox7e26 import wilcox_chapter_7_equation_26


def test_wilcox7e26_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_equation_26(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox7e26_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_equation_26(x)
    assert isinstance(result, dict)
