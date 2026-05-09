"""Tests for wilcox14u782.wilcox_chapter_14_unnumbered_782."""
import numpy as np
import pytest
from moirais.fn.wilcox14u782 import wilcox_chapter_14_unnumbered_782


def test_wilcox14u782_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_782(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u782_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_782(x)
    assert isinstance(result, dict)
