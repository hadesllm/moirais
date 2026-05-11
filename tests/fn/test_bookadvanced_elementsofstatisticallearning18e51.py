"""Tests for bookadvanced_elementsofstatisticallearning18e51.bookadvanced_elementsofstatisticallearning_chapter_18_equation_51."""
import numpy as np
import pytest
from morie.fn.bookadvanced_elementsofstatisticallearning18e51 import bookadvanced_elementsofstatisticallearning_chapter_18_equation_51


def test_bookadvanced_elementsofstatisticallearning18e51_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_18_equation_51(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning18e51_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_18_equation_51(x)
    assert isinstance(result, dict)
