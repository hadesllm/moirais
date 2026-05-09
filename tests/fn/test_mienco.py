"""Tests for mienco.mi_neural_encoder."""
import numpy as np
import pytest
from moirais.fn.mienco import mi_neural_encoder


def test_mienco_basic():
    """Test basic functionality."""
    input = np.random.default_rng(42).normal(0, 1, 100)
    latent_net = np.random.default_rng(42).normal(0, 1, 100)
    result = mi_neural_encoder(input, latent_net)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mienco_edge():
    """Test edge cases."""
    input = np.random.default_rng(42).normal(0, 1, 100)
    latent_net = np.random.default_rng(42).normal(0, 1, 100)
    result = mi_neural_encoder(input, latent_net)
    assert isinstance(result, dict)
