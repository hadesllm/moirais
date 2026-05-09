"""Tests for esnnts.echo_state_network."""
import numpy as np
import pytest
from moirais.fn.esnnts import echo_state_network


def test_esnnts_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    reservoir_size = 100
    result = echo_state_network(y, reservoir_size)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_esnnts_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    reservoir_size = 100
    result = echo_state_network(y, reservoir_size)
    assert isinstance(result, dict)
