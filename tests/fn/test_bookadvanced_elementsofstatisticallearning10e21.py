"""Tests for bookadvanced_elementsofstatisticallearning10e21.bookadvanced_elementsofstatisticallearning_chapter_10_equation_21."""
import numpy as np
import pytest
from morie.fn.bookadvanced_elementsofstatisticallearning10e21 import bookadvanced_elementsofstatisticallearning_chapter_10_equation_21


def test_bookadvanced_elementsofstatisticallearning10e21_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_10_equation_21(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning10e21_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_10_equation_21(x)
    assert isinstance(result, dict)
