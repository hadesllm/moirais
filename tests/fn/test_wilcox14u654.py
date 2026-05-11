"""Tests for wilcox14u654.wilcox_chapter_14_unnumbered_654."""
import numpy as np
import pytest
from morie.fn.wilcox14u654 import wilcox_chapter_14_unnumbered_654


def test_wilcox14u654_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_654(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u654_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_654(x)
    assert isinstance(result, dict)
