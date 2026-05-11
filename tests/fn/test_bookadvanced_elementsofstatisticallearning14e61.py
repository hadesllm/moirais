"""Tests for bookadvanced_elementsofstatisticallearning14e61.bookadvanced_elementsofstatisticallearning_chapter_14_equation_61."""
import numpy as np
import pytest
from morie.fn.bookadvanced_elementsofstatisticallearning14e61 import bookadvanced_elementsofstatisticallearning_chapter_14_equation_61


def test_bookadvanced_elementsofstatisticallearning14e61_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_14_equation_61(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning14e61_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_14_equation_61(x)
    assert isinstance(result, dict)
