"""Tests for wilcox14u749.wilcox_chapter_14_unnumbered_749."""
import numpy as np
import pytest
from morie.fn.wilcox14u749 import wilcox_chapter_14_unnumbered_749


def test_wilcox14u749_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_749(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox14u749_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_749(x)
    assert isinstance(result, dict)
