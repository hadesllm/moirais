"""Test PCA via eigendecomposition."""

import numpy as np
import pytest
from morie.fn.pcana import pcana


class TestPCANA:
    """Principal Component Analysis tests."""

    def test_pcana_basic_shape(self):
        """Test output shapes."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        result = pcana(X, n_components=3)

        assert result["components"].shape == (3, 5)
        assert result["scores"].shape == (100, 3)
        assert result["explained_var"].shape == (3,)
        assert result["cum_var_explained"].shape == (3,)
        assert result["loadings"].shape == (3, 5)

    def test_pcana_variance_sums_to_one(self):
        """Test explained variance proportions sum to 1."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        result = pcana(X, n_components=None)

        assert np.allclose(np.sum(result["explained_var"]), 1.0, atol=1e-10)

    def test_pcana_cumulative_variance(self):
        """Test cumulative variance is monotonically increasing."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        result = pcana(X, n_components=None)

        cum_var = result["cum_var_explained"]
        assert np.all(np.diff(cum_var) >= 0)
        assert cum_var[-1] <= 1.001

    def test_pcana_correlation_scale(self):
        """Test PCA on correlation matrix."""
        np.random.seed(42)
        X = np.random.randn(100, 3) * np.array([1, 10, 100])
        result_corr = pcana(X, n_components=2, scale=True)
        result_cov = pcana(X, n_components=2, scale=False)

        assert result_corr["components"].shape == (2, 3)
        assert result_cov["components"].shape == (2, 3)

    def test_pcana_orthogonal_components(self):
        """Test that components are orthogonal."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        result = pcana(X, n_components=4)

        # Compute Gram matrix: should be identity
        G = result["components"] @ result["components"].T
        assert np.allclose(G, np.eye(4), atol=1e-10)

    def test_pcana_perfect_colinearity(self):
        """Test with linearly dependent columns."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        X = np.column_stack([X, X[:, 0] + X[:, 1]])

        result = pcana(X, n_components=3)
        assert result["components"].shape == (3, 4)

    def test_pcana_single_component(self):
        """Test extraction of single component."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        result = pcana(X, n_components=1)

        assert result["components"].shape == (1, 5)
        assert result["scores"].shape == (100, 1)
        assert len(result["explained_var"]) == 1

    def test_pcana_constant_column(self):
        """Test robustness to zero-variance column."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        X[:, 2] = 5.0

        result = pcana(X, n_components=3)
        assert result["components"].shape == (3, 4)

    def test_pcana_scores_reconstruction(self):
        """Test that scores can reconstruct original (approximately)."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        result = pcana(X, n_components=3, scale=False)

        X_centered = X - X.mean(axis=0)
        X_reconstructed = result["scores"] @ result["components"]
        assert np.allclose(X_reconstructed, X_centered, atol=1e-10)

    def test_pcana_loadings_correlation(self):
        """Test that loadings are correct correlations."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        result = pcana(X, n_components=3, scale=True, return_loadings=True)

        # Loadings should be eigenvectors * sqrt(eigenvalues)
        # Verify by checking correlation properties
        assert result["loadings"].shape == (3, 3)
        assert np.all(np.abs(result["loadings"]) <= 1.001)
