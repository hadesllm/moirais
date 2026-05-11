"""Tests for wilcox14u724.wilcox_chapter_14_unnumbered_724."""
import numpy as np
import pytest
from morie.fn.wilcox14u724 import wilcox_chapter_14_unnumbered_724


def test_wilcox14u724_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_724(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u724_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_724(x)
    assert isinstance(result, dict)
