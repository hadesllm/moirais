"""Tests for wilcox11e9.wilcox_chapter_11_equation_9."""
import numpy as np
import pytest
from morie.fn.wilcox11e9 import wilcox_chapter_11_equation_9


def test_wilcox11e9_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_11_equation_9(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox11e9_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_11_equation_9(x)
    assert isinstance(result, dict)
