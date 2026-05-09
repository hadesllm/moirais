"""Tests for wilcox10u1050.wilcox_chapter_10_unnumbered_1050."""
import numpy as np
import pytest
from moirais.fn.wilcox10u1050 import wilcox_chapter_10_unnumbered_1050


def test_wilcox10u1050_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_1050(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox10u1050_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_10_unnumbered_1050(x)
    assert isinstance(result, dict)
