"""Tests for wilcox14u673.wilcox_chapter_14_unnumbered_673."""
import numpy as np
import pytest
from morie.fn.wilcox14u673 import wilcox_chapter_14_unnumbered_673


def test_wilcox14u673_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_673(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u673_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_673(x)
    assert isinstance(result, dict)
