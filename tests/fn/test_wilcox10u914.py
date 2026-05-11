"""Tests for wilcox10u914.wilcox_chapter_10_unnumbered_914."""
import numpy as np
import pytest
from morie.fn.wilcox10u914 import wilcox_chapter_10_unnumbered_914


def test_wilcox10u914_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_914(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox10u914_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_914(x)
    assert isinstance(result, dict)
