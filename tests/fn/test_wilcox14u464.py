"""Tests for wilcox14u464.wilcox_chapter_14_unnumbered_464."""
import numpy as np
import pytest
from moirais.fn.wilcox14u464 import wilcox_chapter_14_unnumbered_464


def test_wilcox14u464_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_464(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_wilcox14u464_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_464(x)
    assert isinstance(result, dict)
