"""Tests for wilcox4u24.wilcox_chapter_4_unnumbered_24."""
import numpy as np
import pytest
from moirais.fn.wilcox4u24 import wilcox_chapter_4_unnumbered_24


def test_wilcox4u24_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_24(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u24_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_24(x)
    assert isinstance(result, dict)
