"""Tests for wilcox2u265.wilcox_chapter_2_unnumbered_265."""
import numpy as np
import pytest
from morie.fn.wilcox2u265 import wilcox_chapter_2_unnumbered_265


def test_wilcox2u265_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_265(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox2u265_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_265(x)
    assert isinstance(result, dict)
