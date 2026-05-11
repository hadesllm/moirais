"""Tests for bkwtie.burkov_weight_tying."""
import numpy as np
import pytest
from morie.fn.bkwtie import burkov_weight_tying


def test_bkwtie_basic():
    """Test basic functionality."""
    h_last = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_weight_tying(h_last, E)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bkwtie_edge():
    """Test edge cases."""
    h_last = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_weight_tying(h_last, E)
    assert isinstance(result, dict)
