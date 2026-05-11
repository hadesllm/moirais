"""Tests for morie.fn.vcthp — victim help seeking."""

import pytest
from morie.fn.vcthp import victim_help_seeking
from morie.fn._containers import CrimeResult


class TestVictimHelpSeeking:
    def test_basic(self):
        r = victim_help_seeking(30, 100)
        assert isinstance(r, CrimeResult)
        assert r.rate == pytest.approx(0.3)

    def test_invalid(self):
        with pytest.raises(ValueError):
            victim_help_seeking(10, 0)
