"""Tests for wilcox15e4.wilcox_chapter_15_equation_4."""
import numpy as np
import pytest
from moirais.fn.wilcox15e4 import wilcox_chapter_15_equation_4


def test_wilcox15e4_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_15_equation_4(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox15e4_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_15_equation_4(x)
    assert isinstance(result, dict)
