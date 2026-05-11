"""Tests for wilcox7u335.wilcox_chapter_7_unnumbered_335."""
import numpy as np
import pytest
from morie.fn.wilcox7u335 import wilcox_chapter_7_unnumbered_335


def test_wilcox7u335_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_unnumbered_335(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox7u335_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_unnumbered_335(x)
    assert isinstance(result, dict)
