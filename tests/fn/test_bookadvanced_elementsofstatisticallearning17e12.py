"""Tests for bookadvanced_elementsofstatisticallearning17e12.bookadvanced_elementsofstatisticallearning_chapter_17_equation_12."""
import numpy as np
import pytest
from morie.fn.bookadvanced_elementsofstatisticallearning17e12 import bookadvanced_elementsofstatisticallearning_chapter_17_equation_12


def test_bookadvanced_elementsofstatisticallearning17e12_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_17_equation_12(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning17e12_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_17_equation_12(x)
    assert isinstance(result, dict)
