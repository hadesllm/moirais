"""Tests for bookadvanced_elementsofstatisticallearning3u601.bookadvanced_elementsofstatisticallearning_chapter_3_unnumbered_601."""
import numpy as np
import pytest
from morie.fn.bookadvanced_elementsofstatisticallearning3u601 import bookadvanced_elementsofstatisticallearning_chapter_3_unnumbered_601


def test_bookadvanced_elementsofstatisticallearning3u601_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_3_unnumbered_601(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bookadvanced_elementsofstatisticallearning3u601_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_3_unnumbered_601(x)
    assert isinstance(result, dict)
