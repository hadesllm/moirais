"""Tests for wilcox10u942.wilcox_chapter_10_unnumbered_942."""
import numpy as np
import pytest
from morie.fn.wilcox10u942 import wilcox_chapter_10_unnumbered_942


def test_wilcox10u942_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_942(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox10u942_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_942(x)
    assert isinstance(result, dict)
