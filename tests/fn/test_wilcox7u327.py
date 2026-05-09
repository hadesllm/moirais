"""Tests for wilcox7u327.wilcox_chapter_7_unnumbered_327."""
import numpy as np
import pytest
from moirais.fn.wilcox7u327 import wilcox_chapter_7_unnumbered_327


def test_wilcox7u327_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_unnumbered_327(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox7u327_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_unnumbered_327(x)
    assert isinstance(result, dict)
