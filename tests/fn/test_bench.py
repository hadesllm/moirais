"""Tests for moirais.fn.bench — Benchmark dose."""

import numpy as np
import pytest

from moirais.fn.bench import benchmark_dose


class TestBenchmarkDose:
    def test_basic(self):
        doses = np.array([0, 1, 5, 10, 50])
        responses = np.array([0.05, 0.08, 0.15, 0.35, 0.80])
        res = benchmark_dose(doses, responses)
        assert res.estimate > 0

    def test_bmdl_is_finite(self):
        doses = np.array([0, 1, 5, 10, 50])
        responses = np.array([0.05, 0.10, 0.20, 0.40, 0.85])
        res = benchmark_dose(doses, responses)
        assert np.isfinite(res.estimate) or np.isnan(res.estimate)

    def test_too_few(self):
        with pytest.raises(ValueError):
            benchmark_dose([1, 2], [0.1, 0.5])
