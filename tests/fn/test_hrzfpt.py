"""Tests for hrzfpt.horowitz_first_passage_time."""
import numpy as np
import pytest
from moirais.fn.hrzfpt import horowitz_first_passage_time


def test_hrzfpt_basic():
    """Test basic functionality."""
    y_panel = np.random.default_rng(42).normal(0, 1, 100)
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    fU_hat = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_first_passage_time(y_panel, threshold, fU_hat)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzfpt_edge():
    """Test edge cases."""
    y_panel = np.random.default_rng(42).normal(0, 1, 100)
    threshold = np.random.default_rng(42).normal(0, 1, 100)
    fU_hat = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_first_passage_time(y_panel, threshold, fU_hat)
    assert isinstance(result, dict)
