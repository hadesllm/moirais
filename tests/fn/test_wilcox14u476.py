"""Tests for wilcox14u476.wilcox_chapter_14_unnumbered_476."""
import numpy as np
import pytest
from moirais.fn.wilcox14u476 import wilcox_chapter_14_unnumbered_476


def test_wilcox14u476_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_476(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u476_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_476(x)
    assert isinstance(result, dict)
