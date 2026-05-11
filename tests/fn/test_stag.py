"""Tests for morie.fn.stag -- Staggered DiD estimator."""

import numpy as np
import pytest
from morie.fn.stag import staggered_did


class TestStaggeredDiD:
    def test_basic_staggered(self):
        """Two cohorts with known effect."""
        rng = np.random.default_rng(42)
        # 10 units, 4 periods. Units 0-4 never treated, 5-7 treated at t=2, 8-9 at t=3
        records = []
        for unit in range(10):
            for t in range(1, 5):
                if unit < 5:
                    ft = 0  # never treated
                elif unit < 8:
                    ft = 2
                else:
                    ft = 3
                y = rng.normal(0, 0.5)
                if ft > 0 and t >= ft:
                    y += 2.0  # treatment effect
                records.append((y, unit, t, ft))
        outcome = np.array([r[0] for r in records])
        group = np.array([r[1] for r in records])
        time = np.array([r[2] for r in records])
        first_treat = np.array([r[3] for r in records])
        result = staggered_did(outcome, group, time, first_treat)
        assert result["att_overall"] > 0.5
        assert result["n_groups"] == 2

    def test_no_never_treated_raises(self):
        with pytest.raises(ValueError, match="never-treated"):
            staggered_did(
                np.ones(10), np.arange(10), np.ones(10, dtype=int),
                np.ones(10, dtype=int)
            )

    def test_att_gt_list_populated(self):
        rng = np.random.default_rng(42)
        n = 60
        group = np.repeat(np.arange(6), 10)
        time = np.tile(np.arange(1, 11), 6)
        first_treat = np.repeat([0, 0, 0, 5, 5, 8], 10)
        outcome = rng.normal(0, 1, n)
        result = staggered_did(outcome, group, time, first_treat)
        assert len(result["att_gt"]) > 0
