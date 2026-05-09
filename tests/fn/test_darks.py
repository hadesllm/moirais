"""Tests for moirais.fn.darks -- Dark energy equation of state."""

import numpy as np
from moirais.fn.darks import dark_energy_eos, darks
from moirais.fn._containers import DescriptiveResult


class TestDarks:
    def test_alias(self):
        assert darks is dark_energy_eos

    def test_lambda_cdm(self):
        result = dark_energy_eos(w0=-1.0, wa=0.0)
        assert isinstance(result, DescriptiveResult)
        assert result.extra["is_lambda_cdm"] is True
        assert all(w == -1.0 for w in result.value["w_z"])

    def test_h0_at_z0(self):
        result = dark_energy_eos(h0=70.0)
        assert abs(result.value["H_z"][0] - 70.0) < 0.1
