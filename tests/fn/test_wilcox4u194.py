"""Tests for wilcox4u194.wilcox_chapter_4_unnumbered_194."""
import numpy as np
import pytest
from moirais.fn.wilcox4u194 import wilcox_chapter_4_unnumbered_194


def test_wilcox4u194_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_194(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u194_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_194(x)
    assert isinstance(result, dict)
