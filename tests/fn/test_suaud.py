"""Tests for morie.fn.suaud -- AUDIT score."""

import pytest
from morie.fn.suaud import audit_score


class TestAuditScore:
    def test_low_risk(self):
        res = audit_score([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        assert res.estimate == 1.0
        assert res.extra["zone"] == "low_risk"

    def test_hazardous(self):
        res = audit_score([2, 2, 2, 1, 1, 0, 0, 0, 0, 0])
        assert res.extra["zone"] == "hazardous"

    def test_wrong_length(self):
        with pytest.raises(ValueError):
            audit_score([1, 2, 3])
