"""Tests for wilcox2u290.wilcox_chapter_2_unnumbered_290."""
import numpy as np
import pytest
from morie.fn.wilcox2u290 import wilcox_chapter_2_unnumbered_290


def test_wilcox2u290_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_290(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox2u290_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_290(x)
    assert isinstance(result, dict)
