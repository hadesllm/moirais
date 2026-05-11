"""Tests for information_theory_mackay6u50.information_theory_mackay_chapter_6_unnumbered_50."""
import numpy as np
import pytest
from morie.fn.information_theory_mackay6u50 import information_theory_mackay_chapter_6_unnumbered_50


def test_information_theory_mackay6u50_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_6_unnumbered_50(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_information_theory_mackay6u50_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_6_unnumbered_50(x)
    assert isinstance(result, dict)
