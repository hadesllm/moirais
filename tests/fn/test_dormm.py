"""Tests for morie.fn.dormm -- recurrence quantification analysis."""

import numpy as np
from morie.fn.dormm import recurrence_quantification, dormm
from morie.fn._containers import DescriptiveResult


class TestDormm:
    def test_alias(self):
        assert dormm is recurrence_quantification

    def test_periodic(self):
        t = np.linspace(0, 4 * np.pi, 200)
        x = np.sin(t)
        r = recurrence_quantification(x, embedding_dim=2, delay=5)
        assert isinstance(r, DescriptiveResult)
        assert r.value["RR"] > 0
        assert r.value["DET"] > 0

    def test_rqa_keys(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        r = recurrence_quantification(x)
        for key in ["RR", "DET", "L", "ENTR", "LAM"]:
            assert key in r.value
