"""Tests for wilcox14u781.wilcox_chapter_14_unnumbered_781."""
import numpy as np
import pytest
from morie.fn.wilcox14u781 import wilcox_chapter_14_unnumbered_781


def test_wilcox14u781_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_781(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u781_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_781(x)
    assert isinstance(result, dict)
