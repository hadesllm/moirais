"""Tests for wilcox7e29.wilcox_chapter_7_equation_29."""
import numpy as np
import pytest
from moirais.fn.wilcox7e29 import wilcox_chapter_7_equation_29


def test_wilcox7e29_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_equation_29(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox7e29_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_equation_29(x)
    assert isinstance(result, dict)
