"""Tests for moirais.fn.rskrd — risk reclassification NRI."""

import pytest
import numpy as np
from moirais.fn.rskrd import risk_reclassification
from moirais.fn._containers import ESRes


class TestRiskReclassification:

    def test_returns_esres(self):
        old = np.array([1, 2, 1, 2, 3])
        new = np.array([2, 2, 2, 3, 3])
        outcomes = np.array([1, 0, 1, 1, 0])
        result = risk_reclassification(old, new, outcomes)
        assert isinstance(result, ESRes)

    def test_no_change_nri_zero(self):
        risk = np.array([1, 2, 3, 1, 2])
        outcomes = np.array([0, 1, 1, 0, 1])
        result = risk_reclassification(risk, risk, outcomes)
        assert result.estimate == pytest.approx(0.0)
