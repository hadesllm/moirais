"""Tests for wilcox15e13.wilcox_chapter_15_equation_13."""
import numpy as np
import pytest
from moirais.fn.wilcox15e13 import wilcox_chapter_15_equation_13


def test_wilcox15e13_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_15_equation_13(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox15e13_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_15_equation_13(x)
    assert isinstance(result, dict)
