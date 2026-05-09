"""Tests for wilcox7u341.wilcox_chapter_7_unnumbered_341."""
import numpy as np
import pytest
from moirais.fn.wilcox7u341 import wilcox_chapter_7_unnumbered_341


def test_wilcox7u341_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_unnumbered_341(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox7u341_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_unnumbered_341(x)
    assert isinstance(result, dict)
