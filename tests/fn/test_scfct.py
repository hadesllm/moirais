"""Tests for morie.fn.scfct — factor scores."""

import numpy as np
import pandas as pd
from morie.fn.scfct import score_factor


class TestScoreFactor:
    def test_returns_ndarray(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        data = mapq_df[items]
        loadings = np.array([[0.8], [0.7], [0.6], [0.5], [0.4]])
        result = score_factor(data, loadings)
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(mapq_df), 1)

    def test_multiple_factors(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith(("EE", "EA", "UA", "ER"))]
        data = mapq_df[items]
        loadings = np.random.default_rng(42).standard_normal((20, 4))
        result = score_factor(data, loadings)
        assert result.shape == (len(mapq_df), 4)

    def test_mismatched_raises(self, mapq_df):
        items = [c for c in mapq_df.columns if c.startswith("EE")]
        data = mapq_df[items]
        bad_loadings = np.ones((3, 1))  # wrong number of items
        with __import__("pytest").raises(ValueError):
            score_factor(data, bad_loadings)

    def test_ndarray(self, rng):
        data = rng.integers(1, 6, size=(50, 4))
        loadings = np.array([[0.8], [0.7], [0.6], [0.5]])
        result = score_factor(data, loadings)
        assert result.shape == (50, 1)
