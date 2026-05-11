"""Tests for bookadvanced_elementsofstatisticallearning8e13.bookadvanced_elementsofstatisticallearning_chapter_8_equation_13."""
import numpy as np
import pytest
from morie.fn.bookadvanced_elementsofstatisticallearning8e13 import bookadvanced_elementsofstatisticallearning_chapter_8_equation_13


def test_bookadvanced_elementsofstatisticallearning8e13_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_8_equation_13(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning8e13_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_8_equation_13(x)
    assert isinstance(result, dict)
