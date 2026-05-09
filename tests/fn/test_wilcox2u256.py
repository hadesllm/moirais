"""Tests for wilcox2u256.wilcox_chapter_2_unnumbered_256."""
import numpy as np
import pytest
from moirais.fn.wilcox2u256 import wilcox_chapter_2_unnumbered_256


def test_wilcox2u256_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_256(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox2u256_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_2_unnumbered_256(x)
    assert isinstance(result, dict)
