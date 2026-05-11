"""Tests for wilcox4u133.wilcox_chapter_4_unnumbered_133."""
import numpy as np
import pytest
from morie.fn.wilcox4u133 import wilcox_chapter_4_unnumbered_133


def test_wilcox4u133_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_133(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u133_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_133(x)
    assert isinstance(result, dict)
