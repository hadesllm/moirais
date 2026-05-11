"""Tests for bookadvanced_elementsofstatisticallearning6u1039.bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_1039."""
import numpy as np
import pytest
from morie.fn.bookadvanced_elementsofstatisticallearning6u1039 import bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_1039


def test_bookadvanced_elementsofstatisticallearning6u1039_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_1039(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning6u1039_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_1039(x)
    assert isinstance(result, dict)
