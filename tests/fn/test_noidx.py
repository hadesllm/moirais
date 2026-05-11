"""Tests for morie.fn.noidx -- noise exposure."""

import pytest
from morie.fn.noidx import noise_exposure


class TestNoiseExposure:
    def test_under_oel(self):
        res = noise_exposure(measurements_dba=[80, 75], durations=[4, 4])
        assert res.extra["exceeds_oel"] is False

    def test_over_oel(self):
        res = noise_exposure([90, 88], [4, 4])
        assert res.extra["exceeds_oel"] is True
