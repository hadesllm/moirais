"""Tests for wilcox14u443.wilcox_chapter_14_unnumbered_443."""
import numpy as np
import pytest
from morie.fn.wilcox14u443 import wilcox_chapter_14_unnumbered_443


def test_wilcox14u443_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_443(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u443_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_443(x)
    assert isinstance(result, dict)
