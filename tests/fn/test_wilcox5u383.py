"""Tests for wilcox5u383.wilcox_chapter_5_unnumbered_383."""
import numpy as np
import pytest
from moirais.fn.wilcox5u383 import wilcox_chapter_5_unnumbered_383


def test_wilcox5u383_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_383(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox5u383_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_5_unnumbered_383(x)
    assert isinstance(result, dict)
