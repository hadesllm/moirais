"""Tests for bookadvanced_elementsofstatisticallearning6u1081.bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_1081."""
import numpy as np
import pytest
from morie.fn.bookadvanced_elementsofstatisticallearning6u1081 import bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_1081


def test_bookadvanced_elementsofstatisticallearning6u1081_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_1081(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning6u1081_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_1081(x)
    assert isinstance(result, dict)
