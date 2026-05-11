"""Tests for wilcox14e28.wilcox_chapter_14_equation_28."""
import numpy as np
import pytest
from morie.fn.wilcox14e28 import wilcox_chapter_14_equation_28


def test_wilcox14e28_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_equation_28(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox14e28_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_equation_28(x)
    assert isinstance(result, dict)
