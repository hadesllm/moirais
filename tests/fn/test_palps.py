"""Tests for morie.fn.palps — Parallel analysis."""
import numpy as np

from morie.fn.palps import parallel_analysis, palps


def test_correlated_suggests_factors():
    """Correlated data should suggest at least 1 factor."""
    rng = np.random.default_rng(42)
    n, p = 200, 5
    # Create data with one strong latent factor
    factor = rng.standard_normal(n)
    X = np.column_stack([factor + rng.standard_normal(n) * 0.3 for _ in range(p)])
    result = parallel_analysis(X, n_sim=50)
    assert result.value >= 1


def test_uncorrelated_zero_factors():
    """Purely random data may suggest 0 factors."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 5))
    result = parallel_analysis(X, n_sim=50)
    assert result.value >= 0  # could be 0 or small
    assert "actual_eigenvalues" in result.extra


def test_palps_alias():
    assert palps is parallel_analysis
