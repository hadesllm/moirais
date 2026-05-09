"""Tests for wilcox4u187.wilcox_chapter_4_unnumbered_187."""
import numpy as np
import pytest
from moirais.fn.wilcox4u187 import wilcox_chapter_4_unnumbered_187


def test_wilcox4u187_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_187(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox4u187_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_4_unnumbered_187(x)
    assert isinstance(result, dict)
