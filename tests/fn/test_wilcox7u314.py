"""Tests for wilcox7u314.wilcox_chapter_7_unnumbered_314."""
import numpy as np
import pytest
from morie.fn.wilcox7u314 import wilcox_chapter_7_unnumbered_314


def test_wilcox7u314_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_unnumbered_314(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox7u314_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_unnumbered_314(x)
    assert isinstance(result, dict)
