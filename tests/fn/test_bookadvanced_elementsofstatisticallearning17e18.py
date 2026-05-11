"""Tests for bookadvanced_elementsofstatisticallearning17e18.bookadvanced_elementsofstatisticallearning_chapter_17_equation_18."""
import numpy as np
import pytest
from morie.fn.bookadvanced_elementsofstatisticallearning17e18 import bookadvanced_elementsofstatisticallearning_chapter_17_equation_18


def test_bookadvanced_elementsofstatisticallearning17e18_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_17_equation_18(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning17e18_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_17_equation_18(x)
    assert isinstance(result, dict)
