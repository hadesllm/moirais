"""Tests for moirais.fn.rnaht -- R0 from next-generation matrix."""

import numpy as np
import pytest
from moirais.fn.rnaht import r_naught_ngm


class TestRNaught:
    def test_sir_model_r0(self):
        F = np.array([[2.0]])
        V = np.array([[1.0]])
        res = r_naught_ngm(F, V)
        assert res["R0"] == pytest.approx(2.0, rel=1e-10)

    def test_two_strain(self):
        F = np.array([[1.5, 0.0], [0.0, 2.5]])
        V = np.array([[1.0, 0.0], [0.0, 1.0]])
        res = r_naught_ngm(F, V)
        assert res["R0"] == pytest.approx(2.5, rel=1e-10)

    def test_ngm_shape(self):
        F = np.array([[1.0, 0.5], [0.3, 0.8]])
        V = np.eye(2)
        res = r_naught_ngm(F, V)
        assert res["ngm"].shape == (2, 2)

    def test_singular_V_raises(self):
        F = np.array([[1.0]])
        V = np.array([[0.0]])
        with pytest.raises(np.linalg.LinAlgError):
            r_naught_ngm(F, V)

    def test_mismatched_raises(self):
        F = np.array([[1.0, 2.0], [3.0, 4.0]])
        V = np.array([[1.0]])
        with pytest.raises(ValueError):
            r_naught_ngm(F, V)

    def test_nonsquare_raises(self):
        F = np.array([[1.0, 2.0]])
        V = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError):
            r_naught_ngm(F, V)
