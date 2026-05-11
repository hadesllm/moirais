"""Tests for morie.fn.srvey -- survey prevalence."""

import numpy as np
import pytest
from morie.fn.srvey import survey_prevalence


class TestSurveyPrevalence:
    def test_unweighted(self):
        """Unweighted: 30 of 100 => prevalence = 0.3."""
        vals = np.array([1] * 30 + [0] * 70)
        res = survey_prevalence(vals)
        assert res.estimate == pytest.approx(0.3)

    def test_weighted(self):
        """Weighted proportion differs from unweighted."""
        vals = np.array([1, 1, 0, 0, 0])
        weights = np.array([5, 5, 1, 1, 1])
        res = survey_prevalence(vals, weights=weights)
        expected = (5 + 5) / (5 + 5 + 1 + 1 + 1)
        assert res.estimate == pytest.approx(expected, rel=0.01)

    def test_deff(self):
        """Design effect should be >= 1 with unequal weights."""
        vals = np.array([1, 0, 1, 0])
        weights = np.array([10, 1, 10, 1])
        res = survey_prevalence(vals, weights=weights)
        assert res.extra["deff"] > 1.0
