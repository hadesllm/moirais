"""Tests for bookadvanced_elementsofstatisticallearning7e24.bookadvanced_elementsofstatisticallearning_chapter_7_equation_24."""
import numpy as np
import pytest
from moirais.fn.bookadvanced_elementsofstatisticallearning7e24 import bookadvanced_elementsofstatisticallearning_chapter_7_equation_24


def test_bookadvanced_elementsofstatisticallearning7e24_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_7_equation_24(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning7e24_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_7_equation_24(x)
    assert isinstance(result, dict)
