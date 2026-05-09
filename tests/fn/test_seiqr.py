"""Tests for moirais.fn.seiqr -- SEIQR compartmental model."""

import numpy as np
import pytest
from moirais.fn.seiqr import seiqr_model


class TestSEIQR:
    def test_conservation(self):
        res = seiqr_model(beta=0.3, sigma=0.2, gamma=0.1, delta=0.05)
        total = res.S + res.E + res.I + res.R + res.extra["Q"]
        np.testing.assert_allclose(total, 1000.0, atol=0.1)

    def test_quarantine_populated(self):
        res = seiqr_model(beta=0.3, sigma=0.2, gamma=0.1, delta=0.1, t_max=200)
        assert np.max(res.extra["Q"]) > 0.1

    def test_r0_with_quarantine(self):
        res = seiqr_model(beta=0.5, sigma=0.2, gamma=0.1, delta=0.15)
        assert res.R0 == pytest.approx(0.5 / (0.1 + 0.15), rel=1e-6)

    def test_model_label(self):
        res = seiqr_model(beta=0.3, sigma=0.2, gamma=0.1, delta=0.05)
        assert res.model == "SEIQR"

    def test_negative_rate_raises(self):
        with pytest.raises(ValueError):
            seiqr_model(beta=-0.1, sigma=0.2, gamma=0.1, delta=0.05)

    def test_high_quarantine_reduces_peak(self):
        low_q = seiqr_model(beta=0.5, sigma=0.2, gamma=0.1, delta=0.01, t_max=300)
        high_q = seiqr_model(beta=0.5, sigma=0.2, gamma=0.1, delta=0.5, t_max=300)
        assert np.max(high_q.I) < np.max(low_q.I)
