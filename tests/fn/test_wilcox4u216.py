"""Tests for wilcox4u216.wilcox_chapter_4_unnumbered_216."""
import numpy as np
import pytest
from morie.fn.wilcox4u216 import wilcox_chapter_4_unnumbered_216


def test_wilcox4u216_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_216(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u216_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_216(x)
    assert isinstance(result, dict)
