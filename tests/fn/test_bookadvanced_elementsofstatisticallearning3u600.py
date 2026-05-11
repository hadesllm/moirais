"""Tests for bookadvanced_elementsofstatisticallearning3u600.bookadvanced_elementsofstatisticallearning_chapter_3_unnumbered_600."""
import numpy as np
import pytest
from morie.fn.bookadvanced_elementsofstatisticallearning3u600 import bookadvanced_elementsofstatisticallearning_chapter_3_unnumbered_600


def test_bookadvanced_elementsofstatisticallearning3u600_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_3_unnumbered_600(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning3u600_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_3_unnumbered_600(x)
    assert isinstance(result, dict)
