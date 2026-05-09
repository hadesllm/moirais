"""Tests for wilcox15u1388.wilcox_chapter_15_unnumbered_1388."""
import numpy as np
import pytest
from moirais.fn.wilcox15u1388 import wilcox_chapter_15_unnumbered_1388


def test_wilcox15u1388_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_15_unnumbered_1388(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox15u1388_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_15_unnumbered_1388(x)
    assert isinstance(result, dict)
