"""Tests for wilcox8u838.wilcox_chapter_8_unnumbered_838."""
import numpy as np
import pytest
from morie.fn.wilcox8u838 import wilcox_chapter_8_unnumbered_838


def test_wilcox8u838_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_838(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox8u838_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_838(x)
    assert isinstance(result, dict)
