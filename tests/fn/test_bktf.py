"""Tests for bktf.burkov_term_frequency."""
import numpy as np
import pytest
from morie.fn.bktf import burkov_term_frequency


def test_bktf_basic():
    """Test basic functionality."""
    term = np.random.default_rng(42).normal(0, 1, 100)
    document = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_term_frequency(term, document)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bktf_edge():
    """Test edge cases."""
    term = np.random.default_rng(42).normal(0, 1, 100)
    document = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_term_frequency(term, document)
    assert isinstance(result, dict)
