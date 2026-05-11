"""Tests for rng169.rangayyan_ch3_abcd_matrix_inversion_lemma."""
import numpy as np
import pytest
from morie.fn.rng169 import rangayyan_ch3_abcd_matrix_inversion_lemma


def test_rng169_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_abcd_matrix_inversion_lemma(A, B, C, D)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng169_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    C = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_abcd_matrix_inversion_lemma(A, B, C, D)
    assert isinstance(result, dict)
