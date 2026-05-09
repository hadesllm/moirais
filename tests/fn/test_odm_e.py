"""Tests for moirais.fn.odm_e — OTIS demo equity."""

import pytest
import numpy as np
from moirais.fn.odm_e import otis_demo_equity
from moirais.fn._containers import ESRes


class TestOtisDemoEquity:

    def test_returns_esres(self):
        gp = np.array([0.5, 0.3, 0.2])
        pp = np.array([0.4, 0.4, 0.2])
        result = otis_demo_equity(gp, pp)
        assert isinstance(result, ESRes)

    def test_equal_proportions(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        result = otis_demo_equity(p, p)
        assert result.estimate == pytest.approx(1.0)
        assert result.extra["dissimilarity_index"] == pytest.approx(0.0)
