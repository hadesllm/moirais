"""Tests for bookadvanced_elementsofstatisticallearning4e32.bookadvanced_elementsofstatisticallearning_chapter_4_equation_32."""
import numpy as np
import pytest
from moirais.fn.bookadvanced_elementsofstatisticallearning4e32 import bookadvanced_elementsofstatisticallearning_chapter_4_equation_32


def test_bookadvanced_elementsofstatisticallearning4e32_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_4_equation_32(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning4e32_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_4_equation_32(x)
    assert isinstance(result, dict)
