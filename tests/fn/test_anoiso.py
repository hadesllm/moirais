"""Tests for morie.fn.anoiso -- isolation forest anomaly detection."""

import numpy as np
from morie.fn.anoiso import anomaly_isolation, anoiso
from morie.fn._containers import DescriptiveResult


class TestAnoiso:
    def test_alias(self):
        assert anoiso is anomaly_isolation

    def test_detects_outliers(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 2))
        X[:10] = 10.0
        r = anomaly_isolation(X, contamination=0.1, n_trees=50)
        assert isinstance(r, DescriptiveResult)
        assert r.extra["n_anomalies"] > 0

    def test_no_crash_1d(self):
        x = np.arange(100, dtype=float)
        r = anomaly_isolation(x, n_trees=10)
        assert len(r.value) == 100
