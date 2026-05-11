"""Tests for wilcox14u770.wilcox_chapter_14_unnumbered_770."""
import numpy as np
import pytest
from morie.fn.wilcox14u770 import wilcox_chapter_14_unnumbered_770


def test_wilcox14u770_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_770(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u770_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_770(x)
    assert isinstance(result, dict)
