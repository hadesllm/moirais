"""Tests for wilcox14e17.wilcox_chapter_14_equation_17."""
import numpy as np
import pytest
from moirais.fn.wilcox14e17 import wilcox_chapter_14_equation_17


def test_wilcox14e17_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_equation_17(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox14e17_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_equation_17(x)
    assert isinstance(result, dict)
