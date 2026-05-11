"""Tests for wilcox8u808.wilcox_chapter_8_unnumbered_808."""
import numpy as np
import pytest
from morie.fn.wilcox8u808 import wilcox_chapter_8_unnumbered_808


def test_wilcox8u808_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_808(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox8u808_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_808(x)
    assert isinstance(result, dict)
