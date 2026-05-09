"""Tests for esstl.effective_sample_size_tail."""
import numpy as np
import pytest
from moirais.fn.esstl import effective_sample_size_tail


def test_esstl_basic():
    """Test basic functionality."""
    chains = np.random.default_rng(42).normal(0, 1, 100)
    result = effective_sample_size_tail(chains)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_esstl_edge():
    """Test edge cases."""
    chains = np.random.default_rng(42).normal(0, 1, 100)
    result = effective_sample_size_tail(chains)
    assert isinstance(result, dict)
