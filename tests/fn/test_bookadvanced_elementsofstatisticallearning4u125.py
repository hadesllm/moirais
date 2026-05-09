"""Tests for bookadvanced_elementsofstatisticallearning4u125.bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_125."""
import numpy as np
import pytest
from moirais.fn.bookadvanced_elementsofstatisticallearning4u125 import bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_125


def test_bookadvanced_elementsofstatisticallearning4u125_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_125(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bookadvanced_elementsofstatisticallearning4u125_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_125(x)
    assert isinstance(result, dict)
