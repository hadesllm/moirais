"""Tests for moirais.fn.sar -- Secondary Attack Rate."""

import pytest
from moirais.fn.sar import secondary_attack_rate


class TestSAR:
    def test_known_sar(self):
        result = secondary_attack_rate(5, 20, index_cases=0)
        assert result["sar"] == pytest.approx(0.25)

    def test_with_index_cases(self):
        """10 contacts, 2 index, 3 secondary -> SAR = 3/8."""
        result = secondary_attack_rate(3, 10, index_cases=2)
        assert result["sar"] == pytest.approx(3 / 8)

    def test_ci_brackets_sar(self):
        result = secondary_attack_rate(10, 50)
        assert result["ci_lower"] < result["sar"] < result["ci_upper"]

    def test_exceeds_susceptible_raises(self):
        with pytest.raises(ValueError, match="exceed"):
            secondary_attack_rate(15, 10, index_cases=0)
