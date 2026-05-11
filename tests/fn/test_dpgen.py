"""Tests for dpgen: Dirichlet process stick-breaking."""

import numpy as np
import pytest

from morie.fn.dpgen import dirichlet_process_gen


def test_dpgen_basic():
    """Test DP stick-breaking construction."""
    result = dirichlet_process_gen(alpha=1.0, cutoff=1e-6)

    assert "weights" in result
    assert "log_weights" in result
    assert result["alpha"] == 1.0
    assert result["n_clusters"] > 0

    # Weights should be positive and sum to ~1
    assert np.all(result["weights"] > 0)
    assert 0.9 < np.sum(result["weights"]) <= 1.0


def test_dpgen_concentration():
    """Test effect of concentration parameter."""
    result_low = dirichlet_process_gen(alpha=0.1, n_clusters=50)
    result_high = dirichlet_process_gen(alpha=10.0, n_clusters=50)

    # Higher alpha should give more clusters with comparable weight
    assert result_high["n_clusters"] > result_low["n_clusters"]


def test_dpgen_log_weights():
    """Test log-weight consistency."""
    result = dirichlet_process_gen(alpha=1.0, n_clusters=10)

    expected_log = np.log(result["weights"])
    np.testing.assert_array_almost_equal(result["log_weights"], expected_log)


def test_dpgen_reproducibility():
    """Test reproducibility with seed."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    result1 = dirichlet_process_gen(alpha=1.0, rng=rng1)
    result2 = dirichlet_process_gen(alpha=1.0, rng=rng2)

    np.testing.assert_array_almost_equal(result1["weights"], result2["weights"])


def test_dpgen_invalid_alpha():
    """Test error handling for invalid alpha."""
    with pytest.raises(ValueError):
        dirichlet_process_gen(alpha=-1.0)

    with pytest.raises(ValueError):
        dirichlet_process_gen(alpha=0.0)


def test_dpgen_cutoff():
    """Test cutoff parameter."""
    result1 = dirichlet_process_gen(alpha=1.0, cutoff=0.01)
    result2 = dirichlet_process_gen(alpha=1.0, cutoff=1e-8)

    # Smaller cutoff should give more clusters
    assert result2["n_clusters"] >= result1["n_clusters"]
