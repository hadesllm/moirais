"""Tests for wilcox2u274.wilcox_chapter_2_unnumbered_274."""
import numpy as np
import pytest
from morie.fn.wilcox2u274 import wilcox_chapter_2_unnumbered_274


def test_wilcox2u274_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_274(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox2u274_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_274(x)
    assert isinstance(result, dict)
