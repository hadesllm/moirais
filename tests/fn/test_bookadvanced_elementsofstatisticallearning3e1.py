"""Tests for bookadvanced_elementsofstatisticallearning3e1.bookadvanced_elementsofstatisticallearning_chapter_3_equation_1."""
import numpy as np
import pytest
from moirais.fn.bookadvanced_elementsofstatisticallearning3e1 import bookadvanced_elementsofstatisticallearning_chapter_3_equation_1


def test_bookadvanced_elementsofstatisticallearning3e1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_3_equation_1(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning3e1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_3_equation_1(x)
    assert isinstance(result, dict)
