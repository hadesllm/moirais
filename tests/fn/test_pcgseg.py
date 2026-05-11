"""Tests for pcg_segment."""
import numpy as np, pytest
from morie.fn.pcgseg import pcg_segment


class TestPcgSegment:
    def test_two_peaks_per_cycle(self):
        fs = 1000.0
        env = np.zeros(4000)
        for i in range(4):
            s1 = 500 * i + 100
            s2 = 500 * i + 300
            env[s1:s1 + 50] = 1.0
            env[s2:s2 + 50] = 0.8
        r = pcg_segment(env, fs, min_gap_ms=80)
        assert r.value >= 2
        assert len(r.extra["s1_indices"]) >= 2
        assert len(r.extra["s2_indices"]) >= 2

    def test_empty_envelope(self):
        r = pcg_segment(np.zeros(100), fs=1000.0)
        assert r.value == 0
        assert r.extra["n_cycles"] == 0

    def test_short_signal(self):
        r = pcg_segment(np.array([0.1, 0.2]), fs=1000.0)
        assert r.value == 0

    def test_name(self):
        r = pcg_segment(np.zeros(100), fs=1000.0)
        assert r.name == "pcg_segment"
