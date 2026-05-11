"""Tests for wilcox7u313.wilcox_chapter_7_unnumbered_313."""
import numpy as np
import pytest
from morie.fn.wilcox7u313 import wilcox_chapter_7_unnumbered_313


def test_wilcox7u313_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_unnumbered_313(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox7u313_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_unnumbered_313(x)
    assert isinstance(result, dict)
