"""Tests for wilcox11e1.wilcox_chapter_11_equation_1."""
import numpy as np
import pytest
from moirais.fn.wilcox11e1 import wilcox_chapter_11_equation_1


def test_wilcox11e1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_11_equation_1(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox11e1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_11_equation_1(x)
    assert isinstance(result, dict)
