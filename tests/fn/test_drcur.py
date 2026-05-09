"""Tests for moirais.fn.drcur — Dose-response curve."""

import numpy as np
import pytest

from moirais.fn.drcur import dose_response_curve


class TestDoseResponseCurve:
    def test_fit_basic(self):
        doses = np.array([0.1, 0.5, 1, 2, 5, 10, 20, 50])
        responses = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95])
        res = dose_response_curve(doses, responses)
        assert res.extra["r_squared"] > 0.8

    def test_ec50_reasonable(self):
        doses = np.array([0.1, 1, 5, 10, 50, 100])
        responses = np.array([0.01, 0.1, 0.3, 0.5, 0.9, 0.95])
        res = dose_response_curve(doses, responses)
        assert 1 < res.extra["ec50"] < 100

    def test_too_few(self):
        with pytest.raises(ValueError):
            dose_response_curve([1, 2], [0.5, 0.8])
