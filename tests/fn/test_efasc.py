"""Tests for morie.fn.efasc -- factor scores."""

import numpy as np
from morie.fn.efasc import efa_scores


class TestEfaScores:

    def test_regression_shape(self, mapq_df):
        items = [c for c in mapq_df.columns if c not in ("gender", "age_group")]
        X = mapq_df[items].to_numpy(dtype=np.float64)
        R = np.corrcoef(X, rowvar=False)
        evals, evecs = np.linalg.eigh(R)
        idx = np.argsort(-evals)
        loadings = evecs[:, idx[:4]] * np.sqrt(np.maximum(evals[idx[:4]], 0))
        scores = efa_scores(mapq_df[items], loadings, method="regression")
        assert scores.shape == (len(mapq_df), 4)

    def test_bartlett_shape(self, mapq_df):
        items = [c for c in mapq_df.columns if c not in ("gender", "age_group")]
        X = mapq_df[items].to_numpy(dtype=np.float64)
        R = np.corrcoef(X, rowvar=False)
        evals, evecs = np.linalg.eigh(R)
        idx = np.argsort(-evals)
        loadings = evecs[:, idx[:4]] * np.sqrt(np.maximum(evals[idx[:4]], 0))
        scores = efa_scores(mapq_df[items], loadings, method="bartlett")
        assert scores.shape == (len(mapq_df), 4)

    def test_scores_finite(self, mapq_df):
        items = [c for c in mapq_df.columns if c not in ("gender", "age_group")]
        X = mapq_df[items].to_numpy(dtype=np.float64)
        R = np.corrcoef(X, rowvar=False)
        evals, evecs = np.linalg.eigh(R)
        idx = np.argsort(-evals)
        loadings = evecs[:, idx[:4]] * np.sqrt(np.maximum(evals[idx[:4]], 0))
        scores = efa_scores(mapq_df[items], loadings)
        assert np.all(np.isfinite(scores))
