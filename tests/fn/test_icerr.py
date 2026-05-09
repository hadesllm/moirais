"""Tests for moirais.fn.icerr -- ICER."""

import numpy as np
import pytest
from moirais.fn.icerr import incremental_cost_effectiveness_ratio


class TestICER:
    def test_basic(self):
        res = incremental_cost_effectiveness_ratio(50000, 30000, 10.0, 8.0)
        assert res["icer"] == pytest.approx(20000 / 2.0)
        assert res["delta_cost"] == pytest.approx(20000)
        assert res["delta_effect"] == pytest.approx(2.0)

    def test_dominant(self):
        res = incremental_cost_effectiveness_ratio(20000, 30000, 12.0, 8.0)
        assert "dominant" in res["quadrant"].lower()

    def test_dominated(self):
        res = incremental_cost_effectiveness_ratio(50000, 30000, 7.0, 8.0)
        assert "dominated" in res["quadrant"].lower()

    def test_zero_effect_diff(self):
        res = incremental_cost_effectiveness_ratio(50000, 30000, 10.0, 10.0)
        assert res["icer"] == np.inf

    def test_bootstrap(self):
        res = incremental_cost_effectiveness_ratio(50000, 30000, 10.0, 8.0,
                                                   n_bootstrap=100, seed=42)
        assert "bootstrap_icers" in res
        assert len(res["bootstrap_icers"]) == 100
