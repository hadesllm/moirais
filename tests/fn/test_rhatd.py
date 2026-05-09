"""Tests for moirais.fn.rhatd -- R-hat diagnostic."""

import numpy as np
from moirais.fn.rhatd import rhat_diagnostic


def test_returns_dict():
    rng = np.random.default_rng(42)
    chains = rng.normal(0, 1, (4, 500))
    result = rhat_diagnostic(chains)
    assert isinstance(result, dict)
    assert "rhat" in result


def test_converged_chains():
    rng = np.random.default_rng(42)
    chains = rng.normal(0, 1, (4, 1000))
    result = rhat_diagnostic(chains)
    assert result["converged"] is True
    assert result["rhat"] < 1.05


def test_divergent_chains():
    rng = np.random.default_rng(42)
    chains = np.array([
        rng.normal(0, 0.1, 500),
        rng.normal(100, 0.1, 500),
    ])
    result = rhat_diagnostic(chains)
    assert result["rhat"] > 1.1


def test_single_chain_split():
    rng = np.random.default_rng(42)
    result = rhat_diagnostic(rng.normal(0, 1, 1000))
    assert result["n_chains"] == 2
    assert result["rhat"] < 1.05


def test_list_input():
    rng = np.random.default_rng(42)
    chains = [rng.normal(0, 1, 500).tolist(), rng.normal(0, 1, 500).tolist()]
    result = rhat_diagnostic(chains)
    assert result["rhat"] < 1.05


def test_w_positive():
    rng = np.random.default_rng(42)
    result = rhat_diagnostic(rng.normal(0, 1, (2, 500)))
    assert result["W"] > 0
