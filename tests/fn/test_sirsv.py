"""Tests for morie.fn.sirsv -- SIRS with vaccination."""

import pytest
from morie.fn.sirsv import sirs_vaccination


class TestSIRSV:
    def test_runs(self):
        res = sirs_vaccination()
        assert res.model == "SIRS-V"
        assert len(res.t) == 1000

    def test_r0(self):
        res = sirs_vaccination(beta=0.3, gamma=0.1)
        assert res.R0 == pytest.approx(3.0)

    def test_invalid(self):
        with pytest.raises(ValueError):
            sirs_vaccination(beta=-1)
