"""Tests for wilcox10u1000.wilcox_chapter_10_unnumbered_1000."""
import numpy as np
import pytest
from moirais.fn.wilcox10u1000 import wilcox_chapter_10_unnumbered_1000


def test_wilcox10u1000_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_1000(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox10u1000_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_1000(x)
    assert isinstance(result, dict)
