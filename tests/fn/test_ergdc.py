"""Tests for moirais.fn.ergdc — ergodic decomposition verification."""

import numpy as np
import pytest

from moirais.fn.ergdc import ergdc


class TestErgdc:
    def test_doubly_stochastic_ergodic(self):
        P = np.array([[0.5, 0.5], [0.5, 0.5]])
        result = ergdc(P)
        assert result["is_ergodic"]
        assert result["is_irreducible"]
        assert result["is_aperiodic"]

    def test_identity_not_ergodic(self):
        P = np.eye(3)
        result = ergdc(P)
        assert not result["is_irreducible"]

    def test_stationary_dist_sums_to_one(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        result = ergdc(P)
        assert np.isclose(result["stationary_dist"].sum(), 1.0)

    def test_stationary_is_eigenvector(self):
        P = np.array([[0.6, 0.4], [0.3, 0.7]])
        result = ergdc(P)
        pi = result["stationary_dist"]
        np.testing.assert_allclose(pi @ P, pi, atol=1e-8)

    def test_periodic_not_aperiodic(self):
        P = np.array([[0.0, 1.0], [1.0, 0.0]])
        result = ergdc(P)
        assert result["is_irreducible"]
        assert not result["is_aperiodic"]
        assert not result["is_ergodic"]

    def test_spectral_gap_positive_ergodic(self):
        P = np.array([[0.8, 0.2], [0.3, 0.7]])
        result = ergdc(P)
        assert result["spectral_gap"] > 0

    def test_invalid_not_square(self):
        with pytest.raises(ValueError):
            ergdc(np.array([[0.5, 0.5]]))

    def test_invalid_rows(self):
        with pytest.raises(ValueError):
            ergdc(np.array([[0.3, 0.3], [0.5, 0.5]]))
