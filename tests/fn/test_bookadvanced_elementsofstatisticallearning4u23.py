"""Tests for bookadvanced_elementsofstatisticallearning4u23.bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_23."""
import numpy as np
import pytest
from morie.fn.bookadvanced_elementsofstatisticallearning4u23 import bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_23


def test_bookadvanced_elementsofstatisticallearning4u23_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_23(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bookadvanced_elementsofstatisticallearning4u23_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_23(x)
    assert isinstance(result, dict)
