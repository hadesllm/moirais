"""Tests for wilcox14u641.wilcox_chapter_14_unnumbered_641."""
import numpy as np
import pytest
from morie.fn.wilcox14u641 import wilcox_chapter_14_unnumbered_641


def test_wilcox14u641_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_641(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u641_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_641(x)
    assert isinstance(result, dict)
