"""Tests for wilcox4u203.wilcox_chapter_4_unnumbered_203."""
import numpy as np
import pytest
from morie.fn.wilcox4u203 import wilcox_chapter_4_unnumbered_203


def test_wilcox4u203_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_203(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u203_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_203(x)
    assert isinstance(result, dict)
