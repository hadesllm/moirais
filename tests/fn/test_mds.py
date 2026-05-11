"""Tests for morie.fn.mds — Classical Multidimensional Scaling."""

import numpy as np
import pytest
from morie.fn.mds import mds
from morie.fn._containers import MdsRes


class TestMds:
    """Tests for MDS."""

    def test_returns_mds_res(self, rng):
        X = rng.standard_normal((50, 4))
        result = mds(X, n_dims=2)
        assert isinstance(result, MdsRes)

    def test_output_dimensions(self, rng):
        X = rng.standard_normal((30, 5))
        result = mds(X, n_dims=3)
        assert result.coordinates.shape == (30, 3)

    def test_known_structure(self, rng):
        """Points in 2D should have low stress when embedded in 2D."""
        X = rng.standard_normal((40, 2))
        result = mds(X, n_dims=2)
        assert result.stress < 0.2

    def test_distance_matrix_input(self, rng):
        """Accept pre-computed distance matrix."""
        from scipy.spatial.distance import pdist, squareform
        X = rng.standard_normal((20, 3))
        D = squareform(pdist(X))
        result = mds(D, n_dims=2, is_distance=True)
        assert result.coordinates.shape == (20, 2)
