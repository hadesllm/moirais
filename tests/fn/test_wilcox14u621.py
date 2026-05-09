"""Tests for wilcox14u621.wilcox_chapter_14_unnumbered_621."""
import numpy as np
import pytest
from moirais.fn.wilcox14u621 import wilcox_chapter_14_unnumbered_621


def test_wilcox14u621_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_621(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u621_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_621(x)
    assert isinstance(result, dict)
