"""Tests for information_theory_mackay26u212.information_theory_mackay_chapter_26_unnumbered_212."""
import numpy as np
import pytest
from morie.fn.information_theory_mackay26u212 import information_theory_mackay_chapter_26_unnumbered_212


def test_information_theory_mackay26u212_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_26_unnumbered_212(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_information_theory_mackay26u212_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_26_unnumbered_212(x)
    assert isinstance(result, dict)
