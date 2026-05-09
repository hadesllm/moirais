"""Tests for moirais.fn.shazm -- Simultaneous hypothesis test."""

import numpy as np
from moirais.fn.shazm import simultaneous_test, shazm
from moirais.fn._containers import TestResult


class TestShazm:
    def test_alias(self):
        assert shazm is simultaneous_test

    def test_bonferroni(self):
        p = [0.01, 0.04, 0.001, 0.8]
        result = simultaneous_test(p, method="bonferroni", alpha=0.05)
        assert isinstance(result, TestResult)
        assert result.extra["n_significant"] <= 4

    def test_bh(self):
        p = [0.001, 0.01, 0.05, 0.1, 0.5]
        result = simultaneous_test(p, method="bh")
        assert result.extra["n_significant"] >= 1

    def test_holm(self):
        p = [0.01, 0.03, 0.05]
        result = simultaneous_test(p, method="holm")
        assert len(result.extra["adjusted_p"]) == 3
