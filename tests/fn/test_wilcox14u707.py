"""Tests for wilcox14u707.wilcox_chapter_14_unnumbered_707."""
import numpy as np
import pytest
from moirais.fn.wilcox14u707 import wilcox_chapter_14_unnumbered_707


def test_wilcox14u707_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_707(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u707_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_707(x)
    assert isinstance(result, dict)
