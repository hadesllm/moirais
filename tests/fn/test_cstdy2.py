"""Tests for moirais.fn.cstdy2 — custody days credit."""

import pytest
import numpy as np
from moirais.fn.cstdy2 import custody_days_credit
from moirais.fn._containers import ESRes


class TestCustodyDaysCredit:

    def test_returns_esres(self):
        days = np.array([10, 20, 30])
        result = custody_days_credit(days)
        assert isinstance(result, ESRes)
        assert result.estimate == pytest.approx(30.0)

    def test_custom_ratio(self):
        days = np.array([100.0])
        result = custody_days_credit(days, credit_ratio=2.0)
        assert result.extra["total_credited"] == pytest.approx(200.0)
