"""Tests for moirais.fn.lda_ — Linear Discriminant Analysis."""

import numpy as np
import pytest
from moirais.fn.lda_ import lda_
from moirais.fn._containers import LdaRes


class TestLda:
    """Tests for Fisher's LDA."""

    def test_returns_lda_res(self, rng):
        X = rng.standard_normal((60, 4))
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)
        result = lda_(X, y)
        assert isinstance(result, LdaRes)

    def test_components_shape(self, rng):
        """3 classes -> max 2 discriminant components."""
        X = rng.standard_normal((90, 5))
        y = np.array([0] * 30 + [1] * 30 + [2] * 30)
        result = lda_(X, y)
        assert result.components.shape == (5, 2)
        assert result.projected.shape == (90, 2)

    def test_separation(self, rng):
        """Well-separated classes should be distinguishable on LD1."""
        c1 = rng.standard_normal((40, 3)) + [5, 0, 0]
        c2 = rng.standard_normal((40, 3)) + [-5, 0, 0]
        X = np.vstack([c1, c2])
        y = np.array([0] * 40 + [1] * 40)
        result = lda_(X, y, n_components=1)
        proj = result.projected.ravel()
        # Group means on LD1 should be well separated
        mean_diff = abs(proj[:40].mean() - proj[40:].mean())
        pooled_std = np.std(proj)
        assert mean_diff / pooled_std > 1.5

    def test_n_components_truncation(self, rng):
        X = rng.standard_normal((60, 4))
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)
        result = lda_(X, y, n_components=1)
        assert result.projected.shape[1] == 1
