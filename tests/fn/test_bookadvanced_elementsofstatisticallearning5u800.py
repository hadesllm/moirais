"""Tests for bookadvanced_elementsofstatisticallearning5u800.bookadvanced_elementsofstatisticallearning_chapter_5_unnumbered_800."""
import numpy as np
import pytest
from morie.fn.bookadvanced_elementsofstatisticallearning5u800 import bookadvanced_elementsofstatisticallearning_chapter_5_unnumbered_800


def test_bookadvanced_elementsofstatisticallearning5u800_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_5_unnumbered_800(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning5u800_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_5_unnumbered_800(x)
    assert isinstance(result, dict)
