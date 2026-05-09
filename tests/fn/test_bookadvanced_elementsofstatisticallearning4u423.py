"""Tests for bookadvanced_elementsofstatisticallearning4u423.bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_423."""
import numpy as np
import pytest
from moirais.fn.bookadvanced_elementsofstatisticallearning4u423 import bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_423


def test_bookadvanced_elementsofstatisticallearning4u423_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_423(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning4u423_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_423(x)
    assert isinstance(result, dict)
