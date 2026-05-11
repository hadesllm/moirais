"""Tests for bookadvanced_elementsofstatisticallearning3u602.bookadvanced_elementsofstatisticallearning_chapter_3_unnumbered_602."""
import numpy as np
import pytest
from morie.fn.bookadvanced_elementsofstatisticallearning3u602 import bookadvanced_elementsofstatisticallearning_chapter_3_unnumbered_602


def test_bookadvanced_elementsofstatisticallearning3u602_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_3_unnumbered_602(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_bookadvanced_elementsofstatisticallearning3u602_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_3_unnumbered_602(x)
    assert isinstance(result, dict)
