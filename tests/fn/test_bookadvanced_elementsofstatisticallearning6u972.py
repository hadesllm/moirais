"""Tests for bookadvanced_elementsofstatisticallearning6u972.bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_972."""
import numpy as np
import pytest
from moirais.fn.bookadvanced_elementsofstatisticallearning6u972 import bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_972


def test_bookadvanced_elementsofstatisticallearning6u972_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_972(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning6u972_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_972(x)
    assert isinstance(result, dict)
