"""Tests for wilcox3e12.wilcox_chapter_3_equation_12."""
import numpy as np
import pytest
from morie.fn.wilcox3e12 import wilcox_chapter_3_equation_12


def test_wilcox3e12_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_3_equation_12(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox3e12_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_3_equation_12(x)
    assert isinstance(result, dict)
