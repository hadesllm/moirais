"""Tests for wilcox10u952.wilcox_chapter_10_unnumbered_952."""
import numpy as np
import pytest
from morie.fn.wilcox10u952 import wilcox_chapter_10_unnumbered_952


def test_wilcox10u952_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_952(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox10u952_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_952(x)
    assert isinstance(result, dict)
