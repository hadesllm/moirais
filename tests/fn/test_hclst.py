"""Tests for morie.fn.hclst — Hierarchical clustering."""

import numpy as np
import pytest
from morie.fn.hclst import hclst
from morie.fn._containers import HclstRes


class TestHclst:
    """Tests for hierarchical agglomerative clustering."""

    def test_returns_hclst_res(self, rng):
        X = rng.standard_normal((40, 3))
        result = hclst(X, k=3)
        assert isinstance(result, HclstRes)

    def test_labels_count(self, rng):
        X = rng.standard_normal((50, 4))
        result = hclst(X, k=4)
        assert len(np.unique(result.labels)) == 4
        assert len(result.labels) == 50

    def test_well_separated(self, rng):
        """Well-separated clusters should be correctly assigned."""
        c1 = rng.standard_normal((20, 2)) + [20, 20]
        c2 = rng.standard_normal((20, 2)) + [-20, -20]
        X = np.vstack([c1, c2])
        result = hclst(X, k=2)
        assert len(np.unique(result.labels[:20])) == 1
        assert len(np.unique(result.labels[20:])) == 1

    def test_linkage_matrix_shape(self, rng):
        X = rng.standard_normal((30, 3))
        result = hclst(X, k=2)
        # Linkage matrix has (n-1) rows, 4 columns
        assert result.linkage_matrix.shape == (29, 4)
