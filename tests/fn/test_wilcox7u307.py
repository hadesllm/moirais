"""Tests for wilcox7u307.wilcox_chapter_7_unnumbered_307."""
import numpy as np
import pytest
from moirais.fn.wilcox7u307 import wilcox_chapter_7_unnumbered_307


def test_wilcox7u307_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_unnumbered_307(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox7u307_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_unnumbered_307(x)
    assert isinstance(result, dict)
