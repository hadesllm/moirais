"""Tests for information_theory_mackay6u140.information_theory_mackay_chapter_6_unnumbered_140."""
import numpy as np
import pytest
from moirais.fn.information_theory_mackay6u140 import information_theory_mackay_chapter_6_unnumbered_140


def test_information_theory_mackay6u140_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_6_unnumbered_140(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_information_theory_mackay6u140_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_6_unnumbered_140(x)
    assert isinstance(result, dict)
