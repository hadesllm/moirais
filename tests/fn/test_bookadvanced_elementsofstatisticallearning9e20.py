"""Tests for bookadvanced_elementsofstatisticallearning9e20.bookadvanced_elementsofstatisticallearning_chapter_9_equation_20."""
import numpy as np
import pytest
from morie.fn.bookadvanced_elementsofstatisticallearning9e20 import bookadvanced_elementsofstatisticallearning_chapter_9_equation_20


def test_bookadvanced_elementsofstatisticallearning9e20_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_9_equation_20(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning9e20_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_9_equation_20(x)
    assert isinstance(result, dict)
