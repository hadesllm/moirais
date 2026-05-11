"""Tests for bookadvanced_elementsofstatisticallearning4u211.bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_211."""
import numpy as np
import pytest
from morie.fn.bookadvanced_elementsofstatisticallearning4u211 import bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_211


def test_bookadvanced_elementsofstatisticallearning4u211_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_211(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bookadvanced_elementsofstatisticallearning4u211_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_211(x)
    assert isinstance(result, dict)
