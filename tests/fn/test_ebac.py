"""Tests for moirais.fn.ebac — Widmark eBAC calculation."""

import pytest
from moirais.fn.ebac import calculate_ebac as ebac


class TestCalculateEbac:
    """Tests for calculate_ebac (Widmark formula)."""

    def test_basic_computation(self):
        """Two drinks, 180 lbs male, 1 hour elapsed."""
        result = ebac(drinks=2, weight_lbs=180, hours=1, gender_constant=0.73)
        # (2 * 5.14) / (180 * 0.73) - 0.015 * 1
        expected = (10.28 / 131.4) - 0.015
        assert result == pytest.approx(expected, abs=1e-6)
        assert result > 0

    def test_zero_weight_returns_zero(self):
        """Zero body weight should return 0.0 (guard clause)."""
        assert ebac(drinks=5, weight_lbs=0, hours=0, gender_constant=0.73) == 0.0

    def test_negative_result_clamped_to_zero(self):
        """Long elapsed time should clamp BAC to zero, not go negative."""
        result = ebac(drinks=1, weight_lbs=200, hours=100, gender_constant=0.73)
        assert result == 0.0

    def test_zero_drinks(self):
        """No drinks consumed should yield zero after metabolism."""
        result = ebac(drinks=0, weight_lbs=180, hours=1, gender_constant=0.73)
        assert result == 0.0

    def test_female_constant_higher_bac(self):
        """Female constant (0.66) should produce higher BAC than male (0.73)."""
        male = ebac(drinks=3, weight_lbs=150, hours=1, gender_constant=0.73)
        female = ebac(drinks=3, weight_lbs=150, hours=1, gender_constant=0.66)
        assert female > male
