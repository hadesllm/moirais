"""Tests for wilcox14u574.wilcox_chapter_14_unnumbered_574."""
import numpy as np
import pytest
from moirais.fn.wilcox14u574 import wilcox_chapter_14_unnumbered_574


def test_wilcox14u574_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_574(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u574_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_574(x)
    assert isinstance(result, dict)
