"""Tests for information_theory_mackay6u129.information_theory_mackay_chapter_6_unnumbered_129."""
import numpy as np
import pytest
from morie.fn.information_theory_mackay6u129 import information_theory_mackay_chapter_6_unnumbered_129


def test_information_theory_mackay6u129_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_6_unnumbered_129(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_information_theory_mackay6u129_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_6_unnumbered_129(x)
    assert isinstance(result, dict)
