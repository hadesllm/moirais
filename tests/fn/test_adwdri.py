"""Tests for morie.fn.adwdri -- ADWIN concept drift detection."""

import numpy as np
from morie.fn.adwdri import adwin_drift, adwdri
from morie.fn._containers import DescriptiveResult


class TestAdwdri:
    def test_alias(self):
        assert adwdri is adwin_drift

    def test_detects_shift(self):
        rng = np.random.default_rng(42)
        stream = np.concatenate([rng.normal(0, 1, 200), rng.normal(5, 1, 200)])
        r = adwin_drift(stream, delta=0.01)
        assert isinstance(r, DescriptiveResult)
        assert r.extra["n_drifts"] >= 1

    def test_stable_stream(self):
        stream = np.ones(200)
        r = adwin_drift(stream, delta=0.001)
        assert r.extra["n_drifts"] == 0
