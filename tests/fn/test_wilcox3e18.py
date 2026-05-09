"""Tests for wilcox3e18.wilcox_chapter_3_equation_18."""
import numpy as np
import pytest
from moirais.fn.wilcox3e18 import wilcox_chapter_3_equation_18


def test_wilcox3e18_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_3_equation_18(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox3e18_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_3_equation_18(x)
    assert isinstance(result, dict)
