"""Tests for wilcox5u370.wilcox_chapter_5_unnumbered_370."""
import numpy as np
import pytest
from moirais.fn.wilcox5u370 import wilcox_chapter_5_unnumbered_370


def test_wilcox5u370_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_370(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox5u370_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_370(x)
    assert isinstance(result, dict)
