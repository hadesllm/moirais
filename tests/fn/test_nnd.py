"""Tests for morie.fn.nnd -- Number Needed to Diagnose."""

import pytest
from morie.fn.nnd import number_needed_to_diagnose


class TestNND:
    def test_perfect_test(self):
        result = number_needed_to_diagnose(sensitivity=1.0, specificity=1.0)
        assert result["nnd"] == pytest.approx(1.0, abs=1e-10)
        assert result["youden_j"] == pytest.approx(1.0, abs=1e-10)

    def test_reasonable_test(self):
        result = number_needed_to_diagnose(sensitivity=0.9, specificity=0.8)
        # J = 0.7, NND = 1/0.7 ~ 1.43
        assert result["nnd"] == pytest.approx(1.0 / 0.7, abs=0.01)

    def test_chance_test_raises(self):
        with pytest.raises(ValueError, match="no better than chance"):
            number_needed_to_diagnose(sensitivity=0.5, specificity=0.5)

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            number_needed_to_diagnose(sensitivity=1.5, specificity=0.8)
