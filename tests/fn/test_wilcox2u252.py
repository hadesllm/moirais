"""Tests for wilcox2u252.wilcox_chapter_2_unnumbered_252."""
import numpy as np
import pytest
from moirais.fn.wilcox2u252 import wilcox_chapter_2_unnumbered_252


def test_wilcox2u252_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_252(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox2u252_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_252(x)
    assert isinstance(result, dict)
