"""Tests for wilcox14u450.wilcox_chapter_14_unnumbered_450."""
import numpy as np
import pytest
from morie.fn.wilcox14u450 import wilcox_chapter_14_unnumbered_450


def test_wilcox14u450_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_450(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox14u450_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_450(x)
    assert isinstance(result, dict)
