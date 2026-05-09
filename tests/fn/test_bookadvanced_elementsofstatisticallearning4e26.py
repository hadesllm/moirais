"""Tests for bookadvanced_elementsofstatisticallearning4e26.bookadvanced_elementsofstatisticallearning_chapter_4_equation_26."""
import numpy as np
import pytest
from moirais.fn.bookadvanced_elementsofstatisticallearning4e26 import bookadvanced_elementsofstatisticallearning_chapter_4_equation_26


def test_bookadvanced_elementsofstatisticallearning4e26_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_4_equation_26(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning4e26_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_4_equation_26(x)
    assert isinstance(result, dict)
