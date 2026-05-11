"""Tests for bookadvanced_elementsofstatisticallearning8e46.bookadvanced_elementsofstatisticallearning_chapter_8_equation_46."""
import numpy as np
import pytest
from morie.fn.bookadvanced_elementsofstatisticallearning8e46 import bookadvanced_elementsofstatisticallearning_chapter_8_equation_46


def test_bookadvanced_elementsofstatisticallearning8e46_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_8_equation_46(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning8e46_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_8_equation_46(x)
    assert isinstance(result, dict)
