"""Tests for moirais.fn.bneop -- birth-death process."""

from moirais.fn.bneop import birth_death_process, bneop
from moirais.fn._containers import DescriptiveResult


class TestBneop:
    def test_alias(self):
        assert bneop is birth_death_process

    def test_growth(self):
        result = birth_death_process(birth_rate=2.0, death_rate=0.5, n0=10, t_max=5, seed=42)
        assert isinstance(result, DescriptiveResult)
        assert result.value > 10

    def test_extinction(self):
        result = birth_death_process(birth_rate=0.01, death_rate=5.0, n0=5, t_max=10, seed=42)
        assert result.value == 0 or result.extra["extinction_time"] is not None
