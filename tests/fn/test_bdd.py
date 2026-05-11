"""Tests for morie.fn.bdd -- Bunching DiD."""

import numpy as np
import pandas as pd
from morie.fn.bdd import bunching_did, bdd
from morie.fn._containers import ESRes


class TestBunchingDiD:
    def test_alias(self):
        assert bdd is bunching_did

    def test_detects_change_in_bunching(self):
        """More bunching in post should give positive delta."""
        rng = np.random.default_rng(42)
        r_pre = rng.normal(0, 2, 300)
        r_post_base = rng.normal(0, 2, 200)
        r_post_bunch = rng.normal(0, 0.1, 100)
        r_post = np.concatenate([r_post_base, r_post_bunch])
        r = np.concatenate([r_pre, r_post])
        p = np.concatenate([np.zeros(300), np.ones(300)])
        df = pd.DataFrame({"running": r, "post": p})
        result = bunching_did(df, cutoff=0.0)
        assert isinstance(result, ESRes)
        assert result.extra["b_post"] > result.extra["b_pre"]

    def test_output_structure(self):
        rng = np.random.default_rng(42)
        r = rng.normal(0, 1, 400)
        p = np.concatenate([np.zeros(200), np.ones(200)])
        df = pd.DataFrame({"running": r, "post": p})
        result = bunching_did(df)
        assert "b_pre" in result.extra
        assert "b_post" in result.extra
        assert result.se >= 0
