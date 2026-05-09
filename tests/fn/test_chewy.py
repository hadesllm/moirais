"""Tests for moirais.fn.chewy -- chi-squared alias ('Let the Wookiee win.')."""

from moirais.fn.chewy import chewy, chi_squared_test
from moirais.fn.chisq import chi_square_test


class TestChewy:
    def test_alias_imports(self):
        assert callable(chewy)

    def test_chewy_is_chi_square_test(self):
        assert chewy is chi_square_test

    def test_chi_squared_test_alias(self):
        assert chi_squared_test is chi_square_test

    def test_produces_result(self):
        result = chewy([10, 20, 30])
        assert "chi2" in result
        assert result["p_value"] >= 0
