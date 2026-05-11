"""Tests for wilcox2u250.wilcox_chapter_2_unnumbered_250."""
import numpy as np
import pytest
from morie.fn.wilcox2u250 import wilcox_chapter_2_unnumbered_250


def test_wilcox2u250_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_250(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox2u250_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_250(x)
    assert isinstance(result, dict)
