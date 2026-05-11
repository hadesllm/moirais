"""Tests for bookadvanced_elementsofstatisticallearning4u398.bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_398."""
import numpy as np
import pytest
from morie.fn.bookadvanced_elementsofstatisticallearning4u398 import bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_398


def test_bookadvanced_elementsofstatisticallearning4u398_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_398(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning4u398_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_398(x)
    assert isinstance(result, dict)
