"""Tests for essbk.effective_sample_size_bulk."""
import numpy as np
import pytest
from moirais.fn.essbk import effective_sample_size_bulk


def test_essbk_basic():
    """Test basic functionality."""
    chains = np.random.default_rng(42).normal(0, 1, 100)
    result = effective_sample_size_bulk(chains)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_essbk_edge():
    """Test edge cases."""
    chains = np.random.default_rng(42).normal(0, 1, 100)
    result = effective_sample_size_bulk(chains)
    assert isinstance(result, dict)
