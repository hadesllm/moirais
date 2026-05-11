"""Tests for bookadvanced_elementsofstatisticallearning4u156.bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_156."""
import numpy as np
import pytest
from morie.fn.bookadvanced_elementsofstatisticallearning4u156 import bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_156


def test_bookadvanced_elementsofstatisticallearning4u156_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_156(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning4u156_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_156(x)
    assert isinstance(result, dict)
