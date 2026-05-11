"""Tests for wilcox5u402.wilcox_chapter_5_unnumbered_402."""
import numpy as np
import pytest
from morie.fn.wilcox5u402 import wilcox_chapter_5_unnumbered_402


def test_wilcox5u402_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_402(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_wilcox5u402_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_402(x)
    assert isinstance(result, dict)
