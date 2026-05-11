"""Tests for wilcox15u1389.wilcox_chapter_15_unnumbered_1389."""
import numpy as np
import pytest
from morie.fn.wilcox15u1389 import wilcox_chapter_15_unnumbered_1389


def test_wilcox15u1389_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_15_unnumbered_1389(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox15u1389_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_15_unnumbered_1389(x)
    assert isinstance(result, dict)
