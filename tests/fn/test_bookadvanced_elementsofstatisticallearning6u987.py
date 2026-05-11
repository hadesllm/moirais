"""Tests for bookadvanced_elementsofstatisticallearning6u987.bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_987."""
import numpy as np
import pytest
from morie.fn.bookadvanced_elementsofstatisticallearning6u987 import bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_987


def test_bookadvanced_elementsofstatisticallearning6u987_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_987(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning6u987_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_987(x)
    assert isinstance(result, dict)
