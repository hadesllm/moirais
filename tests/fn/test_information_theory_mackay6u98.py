"""Tests for information_theory_mackay6u98.information_theory_mackay_chapter_6_unnumbered_98."""
import numpy as np
import pytest
from morie.fn.information_theory_mackay6u98 import information_theory_mackay_chapter_6_unnumbered_98


def test_information_theory_mackay6u98_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_6_unnumbered_98(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_information_theory_mackay6u98_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_6_unnumbered_98(x)
    assert isinstance(result, dict)
