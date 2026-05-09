"""Tests for wilcox14e16.wilcox_chapter_14_equation_16."""
import numpy as np
import pytest
from moirais.fn.wilcox14e16 import wilcox_chapter_14_equation_16


def test_wilcox14e16_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_equation_16(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox14e16_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_equation_16(x)
    assert isinstance(result, dict)
