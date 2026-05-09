"""Tests for moirais.fn.awgnc — AWGN channel capacity."""

import numpy as np
import pytest

from moirais.fn.awgnc import awgnc


class TestAwgnc:
    def test_zero_snr(self):
        result = awgnc(snr_linear=0.0)
        assert result["capacity"] == pytest.approx(0.0)

    def test_snr_1_linear(self):
        result = awgnc(snr_linear=1.0)
        assert result["capacity"] == pytest.approx(0.5, abs=1e-10)

    def test_snr_0db(self):
        result = awgnc(snr_db=0.0)
        assert result["capacity"] == pytest.approx(0.5, abs=1e-10)

    def test_snr_10db(self):
        result = awgnc(snr_db=10.0)
        expected = 0.5 * np.log2(1.0 + 10.0)
        assert result["capacity"] == pytest.approx(expected, abs=1e-6)

    def test_both_params_error(self):
        with pytest.raises(ValueError):
            awgnc(snr_db=0.0, snr_linear=1.0)

    def test_no_params_error(self):
        with pytest.raises(ValueError):
            awgnc()

    def test_negative_snr_linear(self):
        with pytest.raises(ValueError):
            awgnc(snr_linear=-1.0)

    def test_db_to_linear_consistency(self):
        result = awgnc(snr_db=3.0)
        assert result["snr_linear"] == pytest.approx(10 ** 0.3, abs=1e-6)
