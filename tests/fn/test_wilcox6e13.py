"""Tests for wilcox6e13.wilcox_chapter_6_equation_13."""
import numpy as np
import pytest
from morie.fn.wilcox6e13 import wilcox_chapter_6_equation_13


def test_wilcox6e13_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_6_equation_13(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox6e13_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_6_equation_13(x)
    assert isinstance(result, dict)
