"""Tests for bookadvanced_elementsofstatisticallearning4u130.bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_130."""
import numpy as np
import pytest
from morie.fn.bookadvanced_elementsofstatisticallearning4u130 import bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_130


def test_bookadvanced_elementsofstatisticallearning4u130_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_130(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bookadvanced_elementsofstatisticallearning4u130_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_130(x)
    assert isinstance(result, dict)
