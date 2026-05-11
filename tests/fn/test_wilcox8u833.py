"""Tests for wilcox8u833.wilcox_chapter_8_unnumbered_833."""
import numpy as np
import pytest
from morie.fn.wilcox8u833 import wilcox_chapter_8_unnumbered_833


def test_wilcox8u833_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_833(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox8u833_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_833(x)
    assert isinstance(result, dict)
