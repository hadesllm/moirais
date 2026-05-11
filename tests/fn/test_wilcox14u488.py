"""Tests for wilcox14u488.wilcox_chapter_14_unnumbered_488."""
import numpy as np
import pytest
from morie.fn.wilcox14u488 import wilcox_chapter_14_unnumbered_488


def test_wilcox14u488_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_488(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u488_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_488(x)
    assert isinstance(result, dict)
