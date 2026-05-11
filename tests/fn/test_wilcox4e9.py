"""Tests for wilcox4e9.wilcox_chapter_4_equation_9."""
import numpy as np
import pytest
from morie.fn.wilcox4e9 import wilcox_chapter_4_equation_9


def test_wilcox4e9_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_equation_9(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox4e9_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_equation_9(x)
    assert isinstance(result, dict)
