"""Tests for morie.fn.vctsv — victim severity."""

import pytest
from morie.fn.vctsv import victim_severity
from morie.fn._containers import ESRes


class TestVictimSeverity:
    def test_basic(self):
        r = victim_severity(["assault", "theft"], {"assault": 10, "theft": 2})
        assert isinstance(r, ESRes)
        assert r.estimate == pytest.approx(6.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            victim_severity([], {})
