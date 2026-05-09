"""Tests for moirais.fn.slepw — Slepian-Wolf bound."""

import numpy as np
import pytest

from moirais.fn.slepw import slepw


class TestSlepw:
    def test_independent_sources(self):
        pxy = np.array([[0.25, 0.25], [0.25, 0.25]])
        result = slepw(pxy)
        assert result["H_X_given_Y"] == pytest.approx(result["H_X"], abs=1e-6)
        assert result["H_Y_given_X"] == pytest.approx(result["H_Y"], abs=1e-6)

    def test_identical_sources(self):
        pxy = np.array([[0.5, 0.0], [0.0, 0.5]])
        result = slepw(pxy)
        assert result["H_X_given_Y"] == pytest.approx(0.0, abs=1e-10)

    def test_chain_rule(self):
        pxy = np.array([[0.3, 0.1], [0.2, 0.4]])
        result = slepw(pxy)
        assert result["H_XY"] == pytest.approx(
            result["H_X"] + result["H_Y_given_X"], abs=1e-10
        )

    def test_sum_rate(self):
        pxy = np.array([[0.4, 0.1], [0.1, 0.4]])
        result = slepw(pxy)
        assert result["R_sum_min"] == pytest.approx(result["H_XY"], abs=1e-10)

    def test_invalid_pmf(self):
        with pytest.raises(ValueError):
            slepw(np.array([[0.3, 0.3], [0.3, 0.3]]))

    def test_1d_error(self):
        with pytest.raises(ValueError):
            slepw(np.array([0.5, 0.5]))
