"""Tests for bookadvanced_elementsofstatisticallearning6u961.bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_961."""
import numpy as np
import pytest
from moirais.fn.bookadvanced_elementsofstatisticallearning6u961 import bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_961


def test_bookadvanced_elementsofstatisticallearning6u961_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_961(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning6u961_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_961(x)
    assert isinstance(result, dict)
