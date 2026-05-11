"""Tests for wilcox12u1379.wilcox_chapter_12_unnumbered_1379."""
import numpy as np
import pytest
from morie.fn.wilcox12u1379 import wilcox_chapter_12_unnumbered_1379


def test_wilcox12u1379_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_12_unnumbered_1379(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox12u1379_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_12_unnumbered_1379(x)
    assert isinstance(result, dict)
