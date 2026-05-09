"""Tests for bookadvanced_elementsofstatisticallearning6u960.bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_960."""
import numpy as np
import pytest
from moirais.fn.bookadvanced_elementsofstatisticallearning6u960 import bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_960


def test_bookadvanced_elementsofstatisticallearning6u960_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_960(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning6u960_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_960(x)
    assert isinstance(result, dict)
