"""Tests for moirais.fn.prs — polygenic risk score."""
import numpy as np
import pytest
from moirais.fn.prs import polygenic_risk_score


class TestPRS:
    def test_basic(self):
        G = np.array([[0, 1, 2], [2, 0, 1], [1, 1, 1]])
        beta = np.array([0.1, 0.2, -0.1])
        res = polygenic_risk_score(G, beta)
        assert res.extra["scores"].shape == (3,)

    def test_pvalue_filter(self):
        G = np.array([[0, 1, 2], [2, 0, 1]])
        beta = np.array([0.5, 0.1, 0.3])
        pv = np.array([1e-9, 0.5, 0.01])
        res = polygenic_risk_score(G, beta, p_values=pv, p_threshold=0.05)
        assert res.extra["n_snps_used"] == 2
