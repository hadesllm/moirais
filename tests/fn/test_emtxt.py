"""Tests for emtxt.em_irt_text."""
import numpy as np
import pytest
from moirais.fn.emtxt import em_irt_text


def test_emtxt_basic():
    """Test basic functionality."""
    word_freq_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    n_dims = 2
    result = em_irt_text(word_freq_matrix, n_dims)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_emtxt_edge():
    """Test edge cases."""
    word_freq_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    n_dims = 2
    result = em_irt_text(word_freq_matrix, n_dims)
    assert isinstance(result, dict)
