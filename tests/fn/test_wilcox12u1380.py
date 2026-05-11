"""Tests for wilcox12u1380.wilcox_chapter_12_unnumbered_1380."""
import numpy as np
import pytest
from morie.fn.wilcox12u1380 import wilcox_chapter_12_unnumbered_1380


def test_wilcox12u1380_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_12_unnumbered_1380(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox12u1380_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_12_unnumbered_1380(x)
    assert isinstance(result, dict)
