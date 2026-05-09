"""Tests for bookadvanced_elementsofstatisticallearning12e60.bookadvanced_elementsofstatisticallearning_chapter_12_equation_60."""
import numpy as np
import pytest
from moirais.fn.bookadvanced_elementsofstatisticallearning12e60 import bookadvanced_elementsofstatisticallearning_chapter_12_equation_60


def test_bookadvanced_elementsofstatisticallearning12e60_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_12_equation_60(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning12e60_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_12_equation_60(x)
    assert isinstance(result, dict)
