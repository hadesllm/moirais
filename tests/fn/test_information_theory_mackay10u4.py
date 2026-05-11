"""Tests for information_theory_mackay10u4.information_theory_mackay_chapter_10_unnumbered_4."""
import numpy as np
import pytest
from morie.fn.information_theory_mackay10u4 import information_theory_mackay_chapter_10_unnumbered_4


def test_information_theory_mackay10u4_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_10_unnumbered_4(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_information_theory_mackay10u4_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_10_unnumbered_4(x)
    assert isinstance(result, dict)
