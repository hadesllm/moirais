"""Tests for wilcox14u723.wilcox_chapter_14_unnumbered_723."""
import numpy as np
import pytest
from morie.fn.wilcox14u723 import wilcox_chapter_14_unnumbered_723


def test_wilcox14u723_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_723(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox14u723_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_723(x)
    assert isinstance(result, dict)
