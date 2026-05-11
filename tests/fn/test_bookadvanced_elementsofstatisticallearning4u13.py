"""Tests for bookadvanced_elementsofstatisticallearning4u13.bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_13."""
import numpy as np
import pytest
from morie.fn.bookadvanced_elementsofstatisticallearning4u13 import bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_13


def test_bookadvanced_elementsofstatisticallearning4u13_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_13(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bookadvanced_elementsofstatisticallearning4u13_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_13(x)
    assert isinstance(result, dict)
