"""Tests for bookadvanced_elementsofstatisticallearning17e11.bookadvanced_elementsofstatisticallearning_chapter_17_equation_11."""
import numpy as np
import pytest
from moirais.fn.bookadvanced_elementsofstatisticallearning17e11 import bookadvanced_elementsofstatisticallearning_chapter_17_equation_11


def test_bookadvanced_elementsofstatisticallearning17e11_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_17_equation_11(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning17e11_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_17_equation_11(x)
    assert isinstance(result, dict)
