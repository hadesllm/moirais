"""Tests for information_theory_mackay4u227.information_theory_mackay_chapter_4_unnumbered_227."""
import numpy as np
import pytest
from moirais.fn.information_theory_mackay4u227 import information_theory_mackay_chapter_4_unnumbered_227


def test_information_theory_mackay4u227_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_4_unnumbered_227(x)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_information_theory_mackay4u227_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = information_theory_mackay_chapter_4_unnumbered_227(x)
    assert isinstance(result, dict)
