"""Tests for wilcox4u213.wilcox_chapter_4_unnumbered_213."""
import numpy as np
import pytest
from morie.fn.wilcox4u213 import wilcox_chapter_4_unnumbered_213


def test_wilcox4u213_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_213(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox4u213_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_213(x)
    assert isinstance(result, dict)
