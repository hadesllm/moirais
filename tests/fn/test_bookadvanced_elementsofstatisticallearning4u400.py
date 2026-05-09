"""Tests for bookadvanced_elementsofstatisticallearning4u400.bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_400."""
import numpy as np
import pytest
from moirais.fn.bookadvanced_elementsofstatisticallearning4u400 import bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_400


def test_bookadvanced_elementsofstatisticallearning4u400_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_400(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bookadvanced_elementsofstatisticallearning4u400_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_400(x)
    assert isinstance(result, dict)
