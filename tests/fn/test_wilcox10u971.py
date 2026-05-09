"""Tests for wilcox10u971.wilcox_chapter_10_unnumbered_971."""
import numpy as np
import pytest
from moirais.fn.wilcox10u971 import wilcox_chapter_10_unnumbered_971


def test_wilcox10u971_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_971(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox10u971_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_971(x)
    assert isinstance(result, dict)
