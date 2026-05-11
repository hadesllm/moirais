"""Tests for morie.fn.crime -- Crime rate per 100K."""

from morie.fn.crime import crime_rate, crime
from morie.fn._containers import CrimeResult


class TestCrime:
    def test_alias(self):
        assert crime is crime_rate

    def test_basic_rate(self):
        result = crime_rate(500, 100_000)
        assert isinstance(result, CrimeResult)
        assert result.rate == 500.0
        assert result.n == 500
        assert result.population == 100_000

    def test_ci_contains_rate(self):
        result = crime_rate(500, 100_000)
        assert result.ci_lower <= result.rate <= result.ci_upper
