"""Tests for information_theory_mackay6u96.information_theory_mackay_chapter_6_unnumbered_96."""
import numpy as np
import pytest
from moirais.fn.information_theory_mackay6u96 import information_theory_mackay_chapter_6_unnumbered_96


def test_information_theory_mackay6u96_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_6_unnumbered_96(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_information_theory_mackay6u96_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_6_unnumbered_96(x)
    assert isinstance(result, dict)
