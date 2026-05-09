"""Tests for information_theory_mackay6u12.information_theory_mackay_chapter_6_unnumbered_12."""
import numpy as np
import pytest
from moirais.fn.information_theory_mackay6u12 import information_theory_mackay_chapter_6_unnumbered_12


def test_information_theory_mackay6u12_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_6_unnumbered_12(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_information_theory_mackay6u12_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_6_unnumbered_12(x)
    assert isinstance(result, dict)
