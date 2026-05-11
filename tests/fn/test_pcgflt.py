"""Tests for pcg_filter."""
import numpy as np, pytest
from morie.fn.pcgflt import pcg_filter


class TestPcgFilter:
    def test_output_length(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 4000)
        r = pcg_filter(x, fs=2000.0)
        assert r.n_samples == 4000
        assert len(r.filtered) == 4000

    def test_attenuates_out_of_band(self):
        fs = 2000.0
        t = np.arange(0, 1, 1 / fs)
        in_band = np.sin(2 * np.pi * 100 * t)
        out_band = np.sin(2 * np.pi * 10 * t)
        x = in_band + out_band
        r = pcg_filter(x, fs)
        power_out = np.mean(r.filtered ** 2)
        power_in = np.mean(in_band ** 2)
        assert power_out < np.mean(x ** 2)
        assert power_out == pytest.approx(power_in, abs=0.1)

    def test_name(self):
        r = pcg_filter(np.zeros(500), fs=1000.0)
        assert r.name == "pcg_filter"
