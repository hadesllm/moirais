"""Tests for bookadvanced_elementsofstatisticallearning4u238.bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_238."""
import numpy as np
import pytest
from morie.fn.bookadvanced_elementsofstatisticallearning4u238 import bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_238


def test_bookadvanced_elementsofstatisticallearning4u238_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_238(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning4u238_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_238(x)
    assert isinstance(result, dict)
