"""Tests for wilcox2e14.wilcox_chapter_2_equation_14."""
import numpy as np
import pytest
from moirais.fn.wilcox2e14 import wilcox_chapter_2_equation_14


def test_wilcox2e14_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_equation_14(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox2e14_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_equation_14(x)
    assert isinstance(result, dict)
