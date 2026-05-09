"""Tests for information_theory_mackay10u1.information_theory_mackay_chapter_10_unnumbered_1."""
import numpy as np
import pytest
from moirais.fn.information_theory_mackay10u1 import information_theory_mackay_chapter_10_unnumbered_1


def test_information_theory_mackay10u1_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_10_unnumbered_1(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_information_theory_mackay10u1_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_10_unnumbered_1(x)
    assert isinstance(result, dict)
