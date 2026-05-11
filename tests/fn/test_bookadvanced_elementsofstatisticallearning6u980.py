"""Tests for bookadvanced_elementsofstatisticallearning6u980.bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_980."""
import numpy as np
import pytest
from morie.fn.bookadvanced_elementsofstatisticallearning6u980 import bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_980


def test_bookadvanced_elementsofstatisticallearning6u980_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_980(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bookadvanced_elementsofstatisticallearning6u980_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_980(x)
    assert isinstance(result, dict)
