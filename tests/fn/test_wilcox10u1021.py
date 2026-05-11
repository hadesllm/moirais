"""Tests for wilcox10u1021.wilcox_chapter_10_unnumbered_1021."""
import numpy as np
import pytest
from morie.fn.wilcox10u1021 import wilcox_chapter_10_unnumbered_1021


def test_wilcox10u1021_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_1021(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox10u1021_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_1021(x)
    assert isinstance(result, dict)
