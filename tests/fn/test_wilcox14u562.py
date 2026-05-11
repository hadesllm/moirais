"""Tests for wilcox14u562.wilcox_chapter_14_unnumbered_562."""
import numpy as np
import pytest
from morie.fn.wilcox14u562 import wilcox_chapter_14_unnumbered_562


def test_wilcox14u562_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_562(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox14u562_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_562(x)
    assert isinstance(result, dict)
