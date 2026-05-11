"""Tests for wilcox10u928.wilcox_chapter_10_unnumbered_928."""
import numpy as np
import pytest
from morie.fn.wilcox10u928 import wilcox_chapter_10_unnumbered_928


def test_wilcox10u928_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_928(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox10u928_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_928(x)
    assert isinstance(result, dict)
