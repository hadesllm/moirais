"""Tests for morie.fn.cmpvl — compliance violation."""

import pytest
import numpy as np
from morie.fn.cmpvl import compliance_violation
from morie.fn._containers import DescriptiveResult


class TestComplianceViolation:

    def test_returns_descriptive(self):
        vt = np.array(["curfew", "drug", "curfew", "assault", "drug"])
        result = compliance_violation(vt)
        assert isinstance(result, DescriptiveResult)
        assert result.extra["n"] == 5

    def test_proportions_sum_one(self):
        vt = np.array(["A", "B", "A", "C"])
        result = compliance_violation(vt)
        assert sum(result.extra["proportions"].values()) == pytest.approx(1.0)
