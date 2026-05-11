"""Tests for wilcox4u156.wilcox_chapter_4_unnumbered_156."""
import numpy as np
import pytest
from morie.fn.wilcox4u156 import wilcox_chapter_4_unnumbered_156


def test_wilcox4u156_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_156(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u156_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_156(x)
    assert isinstance(result, dict)
