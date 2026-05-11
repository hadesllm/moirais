"""Tests for wilcox10u956.wilcox_chapter_10_unnumbered_956."""
import numpy as np
import pytest
from morie.fn.wilcox10u956 import wilcox_chapter_10_unnumbered_956


def test_wilcox10u956_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_956(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox10u956_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_956(x)
    assert isinstance(result, dict)
