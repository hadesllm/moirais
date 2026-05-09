"""Tests for moirais.fn.heceac -- CEAC."""

import numpy as np
from moirais.fn.heceac import ceac


class TestCEAC:
    def test_basic(self):
        rng = np.random.default_rng(42)
        res = ceac(
            cost_diffs=rng.normal(1000, 500, 500),
            effect_diffs=rng.normal(0.5, 0.2, 500),
            wtp_range=[0, 50000, 100000],
        )
        assert res.name == "CEAC"
        assert len(res.value["probability"]) == 3

    def test_monotonic(self):
        rng = np.random.default_rng(42)
        res = ceac(
            rng.normal(1000, 200, 1000),
            rng.normal(0.5, 0.1, 1000),
            wtp_range=list(range(0, 100001, 10000)),
        )
        probs = res.value["probability"]
        assert probs[-1] >= probs[0]
