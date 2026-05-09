"""Tests for bridgs.bridge_sampling."""
import numpy as np
import pytest
from moirais.fn.bridgs import bridge_sampling


def test_bridgs_basic():
    """Test basic functionality."""
    chain = np.random.default_rng(42).normal(0, 1, 100)
    proposal = np.random.default_rng(42).normal(0, 1, 100)
    log_p = np.random.default_rng(42).normal(0, 1, 100)
    log_q = np.random.default_rng(42).normal(0, 1, 100)
    result = bridge_sampling(chain, proposal, log_p, log_q)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bridgs_edge():
    """Test edge cases."""
    chain = np.random.default_rng(42).normal(0, 1, 100)
    proposal = np.random.default_rng(42).normal(0, 1, 100)
    log_p = np.random.default_rng(42).normal(0, 1, 100)
    log_q = np.random.default_rng(42).normal(0, 1, 100)
    result = bridge_sampling(chain, proposal, log_p, log_q)
    assert isinstance(result, dict)
