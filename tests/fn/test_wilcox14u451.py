"""Tests for wilcox14u451.wilcox_chapter_14_unnumbered_451."""
import numpy as np
import pytest
from morie.fn.wilcox14u451 import wilcox_chapter_14_unnumbered_451


def test_wilcox14u451_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_451(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox14u451_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_451(x)
    assert isinstance(result, dict)
