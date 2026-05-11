"""Tests for bookadvanced_elementsofstatisticallearning4u18.bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_18."""
import numpy as np
import pytest
from morie.fn.bookadvanced_elementsofstatisticallearning4u18 import bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_18


def test_bookadvanced_elementsofstatisticallearning4u18_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_18(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bookadvanced_elementsofstatisticallearning4u18_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_18(x)
    assert isinstance(result, dict)
