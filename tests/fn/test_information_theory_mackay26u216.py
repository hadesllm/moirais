"""Tests for information_theory_mackay26u216.information_theory_mackay_chapter_26_unnumbered_216."""
import numpy as np
import pytest
from morie.fn.information_theory_mackay26u216 import information_theory_mackay_chapter_26_unnumbered_216


def test_information_theory_mackay26u216_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_26_unnumbered_216(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_information_theory_mackay26u216_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_26_unnumbered_216(x)
    assert isinstance(result, dict)
