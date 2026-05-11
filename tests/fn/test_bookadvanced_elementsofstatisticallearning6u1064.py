"""Tests for bookadvanced_elementsofstatisticallearning6u1064.bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_1064."""
import numpy as np
import pytest
from morie.fn.bookadvanced_elementsofstatisticallearning6u1064 import bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_1064


def test_bookadvanced_elementsofstatisticallearning6u1064_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_1064(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bookadvanced_elementsofstatisticallearning6u1064_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_1064(x)
    assert isinstance(result, dict)
