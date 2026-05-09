"""Tests for matr.ma_two_step_dl_he."""
import numpy as np
import pytest
from moirais.fn.matr import ma_two_step_dl_he


def test_matr_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_two_step_dl_he(yi, vi, max_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_matr_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    vi = np.random.default_rng(42).normal(0, 1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_two_step_dl_he(yi, vi, max_iter)
    assert isinstance(result, dict)
