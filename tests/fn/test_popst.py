"""Tests for morie.fn.popst -- Population structure PCA."""

import numpy as np
import pytest
from morie.fn.popst import popst


class TestPopst:
    def test_two_populations(self):
        rng = np.random.default_rng(42)
        Z1 = rng.choice([0, 1], size=(20, 50), p=[0.3, 0.7]).astype(float) * 2
        Z2 = rng.choice([0, 1], size=(20, 50), p=[0.7, 0.3]).astype(float) * 2
        Z = np.vstack([Z1, Z2])
        res = popst(Z, n_components=2)
        assert res.statistic > 0

    def test_scores_shape(self):
        rng = np.random.default_rng(42)
        Z = rng.choice([0, 1, 2], size=(15, 30)).astype(float)
        res = popst(Z, n_components=3)
        scores = np.array(res.extra["scores"])
        assert scores.shape == (15, 3)

    def test_var_explained_sums(self):
        rng = np.random.default_rng(42)
        Z = rng.choice([0, 1, 2], size=(10, 20)).astype(float)
        res = popst(Z, n_components=10)
        ve = np.array(res.extra["var_explained"])
        assert np.sum(ve) <= 1.0 + 1e-6

    def test_too_few_individuals(self):
        with pytest.raises(ValueError):
            popst(np.array([[0, 1, 2]]))
