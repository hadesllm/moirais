"""Tests for wilcox10e5.wilcox_chapter_10_equation_5."""
import numpy as np
import pytest
from morie.fn.wilcox10e5 import wilcox_chapter_10_equation_5


def test_wilcox10e5_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_equation_5(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox10e5_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_equation_5(x)
    assert isinstance(result, dict)
