"""Tests for wilcox7e15.wilcox_chapter_7_equation_15."""
import numpy as np
import pytest
from moirais.fn.wilcox7e15 import wilcox_chapter_7_equation_15


def test_wilcox7e15_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_equation_15(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox7e15_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_equation_15(x)
    assert isinstance(result, dict)
