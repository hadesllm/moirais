"""Tests for wilcox7u350.wilcox_chapter_7_unnumbered_350."""
import numpy as np
import pytest
from moirais.fn.wilcox7u350 import wilcox_chapter_7_unnumbered_350


def test_wilcox7u350_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_unnumbered_350(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wilcox7u350_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wilcox_chapter_7_unnumbered_350(x)
    assert isinstance(result, dict)
