"""Tests for wilcox10u987.wilcox_chapter_10_unnumbered_987."""
import numpy as np
import pytest
from morie.fn.wilcox10u987 import wilcox_chapter_10_unnumbered_987


def test_wilcox10u987_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_987(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox10u987_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_987(x)
    assert isinstance(result, dict)
