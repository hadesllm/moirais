"""Tests for morie.fn.stscm -- Byzantine fault detection."""

import numpy as np
from morie.fn.stscm import byzantine_detect, stscm
from morie.fn._containers import DescriptiveResult


class TestStscm:
    def test_alias(self):
        assert stscm is byzantine_detect

    def test_detects_faulty(self):
        rng = np.random.default_rng(42)
        reports = rng.normal(10, 0.5, (10, 20))
        reports[0] = 100.0
        r = byzantine_detect(reports, threshold=2.0)
        assert isinstance(r, DescriptiveResult)
        assert r.value[0] == True

    def test_all_honest(self):
        reports = np.ones((5, 10))
        r = byzantine_detect(reports)
        assert r.extra["n_detected"] == 0
