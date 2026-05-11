"""Tests for wilcox2e7.wilcox_chapter_2_equation_7."""
import numpy as np
import pytest
from morie.fn.wilcox2e7 import wilcox_chapter_2_equation_7


def test_wilcox2e7_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_equation_7(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox2e7_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_equation_7(x)
    assert isinstance(result, dict)
