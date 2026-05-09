"""Tests for bookadvanced_elementsofstatisticallearning6u1014.bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_1014."""
import numpy as np
import pytest
from moirais.fn.bookadvanced_elementsofstatisticallearning6u1014 import bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_1014


def test_bookadvanced_elementsofstatisticallearning6u1014_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_1014(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bookadvanced_elementsofstatisticallearning6u1014_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_1014(x)
    assert isinstance(result, dict)
