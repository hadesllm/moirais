"""Tests for bookadvanced_elementsofstatisticallearning6u1024.bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_1024."""
import numpy as np
import pytest
from moirais.fn.bookadvanced_elementsofstatisticallearning6u1024 import bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_1024


def test_bookadvanced_elementsofstatisticallearning6u1024_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_1024(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning6u1024_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_1024(x)
    assert isinstance(result, dict)
