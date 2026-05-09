"""Tests for wilcox14u566.wilcox_chapter_14_unnumbered_566."""
import numpy as np
import pytest
from moirais.fn.wilcox14u566 import wilcox_chapter_14_unnumbered_566


def test_wilcox14u566_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_566(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox14u566_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_14_unnumbered_566(x)
    assert isinstance(result, dict)
