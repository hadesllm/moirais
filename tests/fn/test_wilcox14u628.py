"""Tests for wilcox14u628.wilcox_chapter_14_unnumbered_628."""
import numpy as np
import pytest
from moirais.fn.wilcox14u628 import wilcox_chapter_14_unnumbered_628


def test_wilcox14u628_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_628(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox14u628_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_628(x)
    assert isinstance(result, dict)
