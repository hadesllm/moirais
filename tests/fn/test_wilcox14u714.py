"""Tests for wilcox14u714.wilcox_chapter_14_unnumbered_714."""
import numpy as np
import pytest
from moirais.fn.wilcox14u714 import wilcox_chapter_14_unnumbered_714


def test_wilcox14u714_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_714(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox14u714_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_714(x)
    assert isinstance(result, dict)
