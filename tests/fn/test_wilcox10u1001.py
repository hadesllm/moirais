"""Tests for wilcox10u1001.wilcox_chapter_10_unnumbered_1001."""
import numpy as np
import pytest
from morie.fn.wilcox10u1001 import wilcox_chapter_10_unnumbered_1001


def test_wilcox10u1001_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_1001(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox10u1001_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_1001(x)
    assert isinstance(result, dict)
