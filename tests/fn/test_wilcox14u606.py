"""Tests for wilcox14u606.wilcox_chapter_14_unnumbered_606."""
import numpy as np
import pytest
from morie.fn.wilcox14u606 import wilcox_chapter_14_unnumbered_606


def test_wilcox14u606_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_606(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox14u606_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_606(x)
    assert isinstance(result, dict)
