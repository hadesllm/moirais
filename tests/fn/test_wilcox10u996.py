"""Tests for wilcox10u996.wilcox_chapter_10_unnumbered_996."""
import numpy as np
import pytest
from morie.fn.wilcox10u996 import wilcox_chapter_10_unnumbered_996


def test_wilcox10u996_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_996(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox10u996_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_996(x)
    assert isinstance(result, dict)
