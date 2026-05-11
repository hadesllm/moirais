"""Tests for morie.fn.kinlr -- Kinship likelihood ratio."""

import numpy as np
import pytest
from morie.fn.kinlr import kinlr


class TestKinlr:
    def test_parent_child(self):
        geno_a = [("A", "B"), ("C", "D")]
        geno_b = [("A", "C"), ("C", "E")]
        freqs = [
            {"A": 0.2, "B": 0.3, "C": 0.3, "D": 0.2},
            {"C": 0.3, "D": 0.2, "E": 0.2, "F": 0.3},
        ]
        res = kinlr(geno_a, geno_b, freqs, relationship="parent_child")
        assert res.extra["LR"] > 0

    def test_unrelated_known_relationship(self):
        geno_a = [("A", "A")]
        geno_b = [("B", "B")]
        freqs = [{"A": 0.5, "B": 0.5}]
        res = kinlr(geno_a, geno_b, freqs, relationship="parent_child")
        assert res.extra["LR"] >= 0

    def test_invalid_relationship(self):
        with pytest.raises(ValueError):
            kinlr([("A", "B")], [("A", "B")], [{"A": 0.5, "B": 0.5}],
                   relationship="cousin")

    def test_mismatched_length(self):
        with pytest.raises(ValueError):
            kinlr([("A", "B")], [("A", "B"), ("C", "D")],
                   [{"A": 0.5, "B": 0.5}])
