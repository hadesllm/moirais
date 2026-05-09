"""Tests for moirais.fn.synds -- Syndromic surveillance."""

import pytest
import numpy as np
from moirais.fn.synds import syndromic_surveillance


class TestSyndromicSurveillance:
    def test_no_alerts(self):
        counts = [10] * 30
        res = syndromic_surveillance(counts, method="C1")
        assert res.estimate == 0

    def test_outbreak(self):
        counts = [10] * 20 + [50, 60, 70]
        res = syndromic_surveillance(counts, method="C1", baseline_days=7)
        assert res.estimate > 0

    def test_too_short(self):
        with pytest.raises(ValueError):
            syndromic_surveillance([1, 2, 3])
