"""Tests for morie.fn.seirc -- SEIR with compartmental dynamics."""

import pytest
from morie.fn.seirc import seir_compartmental


class TestSEIRC:
    def test_runs(self):
        res = seir_compartmental()
        assert res.model == "SEIR-C"
        assert len(res.t) == 1000

    def test_r0(self):
        res = seir_compartmental(beta=0.3, sigma=0.2, gamma=0.1, mu=0.0, omega=0.0)
        assert res.R0 == pytest.approx(0.3 * 0.2 / (0.2 * 0.1), rel=0.01)

    def test_invalid(self):
        with pytest.raises(ValueError):
            seir_compartmental(beta=-1)
