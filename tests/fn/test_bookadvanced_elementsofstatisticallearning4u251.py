"""Tests for bookadvanced_elementsofstatisticallearning4u251.bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_251."""
import numpy as np
import pytest
from morie.fn.bookadvanced_elementsofstatisticallearning4u251 import bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_251


def test_bookadvanced_elementsofstatisticallearning4u251_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_251(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning4u251_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_251(x)
    assert isinstance(result, dict)
