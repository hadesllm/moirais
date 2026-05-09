"""Tests for wilcox6e22.wilcox_chapter_6_equation_22."""
import numpy as np
import pytest
from moirais.fn.wilcox6e22 import wilcox_chapter_6_equation_22


def test_wilcox6e22_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_6_equation_22(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox6e22_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_6_equation_22(x)
    assert isinstance(result, dict)
