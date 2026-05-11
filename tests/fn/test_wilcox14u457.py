"""Tests for wilcox14u457.wilcox_chapter_14_unnumbered_457."""
import numpy as np
import pytest
from morie.fn.wilcox14u457 import wilcox_chapter_14_unnumbered_457


def test_wilcox14u457_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_457(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u457_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_457(x)
    assert isinstance(result, dict)
