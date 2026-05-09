"""Tests for moirais.fn.pca — Principal Component Analysis."""

import numpy as np
import pandas as pd
import pytest
from moirais.fn.pca import pca
from moirais.fn._containers import PcaRes


class TestPca:
    """Tests for PCA."""

    def test_returns_pca_res(self, rng):
        X = rng.standard_normal((100, 5))
        result = pca(X)
        assert isinstance(result, PcaRes)

    def test_full_variance_explained(self, rng):
        """All components should explain 100% of variance."""
        X = rng.standard_normal((100, 4))
        result = pca(X)
        assert abs(np.sum(result.explained_variance_ratio) - 1.0) < 1e-6

    def test_n_components_truncation(self, rng):
        X = rng.standard_normal((100, 6))
        result = pca(X, n_components=2)
        assert result.scores.shape == (100, 2)
        assert result.components.shape == (6, 2)
        assert len(result.explained_variance) == 2

    def test_correlated_data_first_component_dominant(self, rng):
        """Data with strong first PC should have high ratio[0]."""
        latent = rng.standard_normal(200)
        X = np.column_stack([
            latent + rng.standard_normal(200) * 0.1,
            latent + rng.standard_normal(200) * 0.1,
            rng.standard_normal(200),
        ])
        result = pca(X, n_components=2)
        assert result.explained_variance_ratio[0] > 0.5

    def test_dataframe_input(self, rng):
        df = pd.DataFrame(rng.standard_normal((50, 3)), columns=["a", "b", "c"])
        result = pca(df, n_components=2)
        assert result.scores.shape == (50, 2)
