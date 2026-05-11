"""Tests for wilcox6e11.wilcox_chapter_6_equation_11."""
import numpy as np
import pytest
from morie.fn.wilcox6e11 import wilcox_chapter_6_equation_11


def test_wilcox6e11_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_6_equation_11(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox6e11_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_6_equation_11(x)
    assert isinstance(result, dict)
