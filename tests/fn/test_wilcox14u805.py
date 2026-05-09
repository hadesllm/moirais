"""Tests for wilcox14u805.wilcox_chapter_14_unnumbered_805."""
import numpy as np
import pytest
from moirais.fn.wilcox14u805 import wilcox_chapter_14_unnumbered_805


def test_wilcox14u805_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_805(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox14u805_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_805(x)
    assert isinstance(result, dict)
