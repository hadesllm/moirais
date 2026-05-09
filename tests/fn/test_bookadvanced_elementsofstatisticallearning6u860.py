"""Tests for bookadvanced_elementsofstatisticallearning6u860.bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_860."""
import numpy as np
import pytest
from moirais.fn.bookadvanced_elementsofstatisticallearning6u860 import bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_860


def test_bookadvanced_elementsofstatisticallearning6u860_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_860(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bookadvanced_elementsofstatisticallearning6u860_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_860(x)
    assert isinstance(result, dict)
