"""Tests for morie.fn.dbscn — DBSCAN clustering."""

import numpy as np
import pytest
from morie.fn.dbscn import dbscn
from morie.fn._containers import DbscnRes


class TestDbscn:
    """Tests for DBSCAN."""

    def test_returns_dbscn_res(self, rng):
        X = rng.standard_normal((50, 2))
        result = dbscn(X, eps=1.5, min_samples=3)
        assert isinstance(result, DbscnRes)

    def test_well_separated_clusters(self, rng):
        """Two tight clusters far apart should be found."""
        c1 = rng.standard_normal((30, 2)) * 0.1 + [10, 10]
        c2 = rng.standard_normal((30, 2)) * 0.1 + [-10, -10]
        X = np.vstack([c1, c2])
        result = dbscn(X, eps=0.5, min_samples=3)
        assert result.n_clusters == 2
        assert result.n_noise == 0

    def test_noise_detection(self, rng):
        """Scattered random data should produce noise points."""
        X = rng.standard_normal((20, 2)) * 10
        result = dbscn(X, eps=0.3, min_samples=5)
        assert result.n_noise > 0
        assert np.any(result.labels == -1)

    def test_single_dense_cluster(self, rng):
        """One tight blob should yield 1 cluster, 0 noise."""
        X = rng.standard_normal((50, 2)) * 0.1
        result = dbscn(X, eps=1.0, min_samples=3)
        assert result.n_clusters == 1
        assert result.n_noise == 0
