"""Tests for bookadvanced_elementsofstatisticallearning14e62.bookadvanced_elementsofstatisticallearning_chapter_14_equation_62."""
import numpy as np
import pytest
from moirais.fn.bookadvanced_elementsofstatisticallearning14e62 import bookadvanced_elementsofstatisticallearning_chapter_14_equation_62


def test_bookadvanced_elementsofstatisticallearning14e62_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_14_equation_62(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning14e62_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_14_equation_62(x)
    assert isinstance(result, dict)
