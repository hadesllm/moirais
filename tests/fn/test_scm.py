"""Tests for morie.fn.scm — structural causal model."""
import numpy as np
import pytest
from morie.fn.scm import structural_causal_model


class TestStructuralCausalModel:
    def test_linear_dag(self):
        rng = np.random.default_rng(42)
        n = 200
        X = rng.standard_normal(n)
        Y = 0.7 * X + rng.standard_normal(n) * 0.3
        Z = 0.5 * Y + rng.standard_normal(n) * 0.3
        data = {"X": X, "Y": Y, "Z": Z}
        edges = [("X", "Y"), ("Y", "Z")]
        res = structural_causal_model(data, edges)
        coeffs = res.extra["structural_coefficients"]
        assert "X->Y" in coeffs
        assert "Y->Z" in coeffs

    def test_empty_edges_raises(self):
        with pytest.raises(ValueError, match="at least one edge"):
            structural_causal_model({"X": np.array([1.0])}, [])

    def test_intervention(self):
        rng = np.random.default_rng(42)
        n = 200
        X = rng.standard_normal(n)
        Y = 0.8 * X + rng.standard_normal(n) * 0.2
        data = {"X": X, "Y": Y}
        edges = [("X", "Y")]
        res = structural_causal_model(data, edges, intervention={"X": 1.0})
        assert "Y" in res.extra["causal_effects"]
