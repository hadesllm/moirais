"""Tests for wilcox14u739.wilcox_chapter_14_unnumbered_739."""
import numpy as np
import pytest
from moirais.fn.wilcox14u739 import wilcox_chapter_14_unnumbered_739


def test_wilcox14u739_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_739(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u739_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_739(x)
    assert isinstance(result, dict)
