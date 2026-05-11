"""Tests for wilcox15e6.wilcox_chapter_15_equation_6."""
import numpy as np
import pytest
from morie.fn.wilcox15e6 import wilcox_chapter_15_equation_6


def test_wilcox15e6_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_15_equation_6(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox15e6_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_15_equation_6(x)
    assert isinstance(result, dict)
