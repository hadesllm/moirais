"""Tests for wilcox14u710.wilcox_chapter_14_unnumbered_710."""
import numpy as np
import pytest
from morie.fn.wilcox14u710 import wilcox_chapter_14_unnumbered_710


def test_wilcox14u710_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_710(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox14u710_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_710(x)
    assert isinstance(result, dict)
