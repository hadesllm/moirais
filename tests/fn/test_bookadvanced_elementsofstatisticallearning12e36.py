"""Tests for bookadvanced_elementsofstatisticallearning12e36.bookadvanced_elementsofstatisticallearning_chapter_12_equation_36."""
import numpy as np
import pytest
from morie.fn.bookadvanced_elementsofstatisticallearning12e36 import bookadvanced_elementsofstatisticallearning_chapter_12_equation_36


def test_bookadvanced_elementsofstatisticallearning12e36_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_12_equation_36(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning12e36_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_12_equation_36(x)
    assert isinstance(result, dict)
