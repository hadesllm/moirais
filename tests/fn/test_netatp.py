"""Tests for netatp.network_attack_tolerance."""
import numpy as np
import pytest
from moirais.fn.netatp import network_attack_tolerance


def test_netatp_basic():
    """Test basic functionality."""
    G = np.eye(10)
    attack_strategy = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = network_attack_tolerance(G, attack_strategy, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_netatp_edge():
    """Test edge cases."""
    G = np.eye(10)
    attack_strategy = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = network_attack_tolerance(G, attack_strategy, k)
    assert isinstance(result, dict)
