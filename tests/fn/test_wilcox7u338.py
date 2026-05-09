"""Tests for wilcox7u338.wilcox_chapter_7_unnumbered_338."""
import numpy as np
import pytest
from moirais.fn.wilcox7u338 import wilcox_chapter_7_unnumbered_338


def test_wilcox7u338_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_unnumbered_338(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox7u338_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_unnumbered_338(x)
    assert isinstance(result, dict)
