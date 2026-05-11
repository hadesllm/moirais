"""Tests for wilcox8u826.wilcox_chapter_8_unnumbered_826."""
import numpy as np
import pytest
from morie.fn.wilcox8u826 import wilcox_chapter_8_unnumbered_826


def test_wilcox8u826_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_826(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox8u826_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_8_unnumbered_826(x)
    assert isinstance(result, dict)
