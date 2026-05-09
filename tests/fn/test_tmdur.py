"""Test time_duration (tmdur)."""
from moirais.fn.tmdur import time_duration, tmdur
from moirais.fn._containers import DescriptiveResult


class TestTmdur:
    def test_basic(self):
        result = time_duration(1000, 500.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "time_duration"
        assert result.value == 2.0

    def test_known_value(self):
        result = time_duration(44100, 44100.0)
        assert result.value == 1.0

    def test_alias(self):
        assert tmdur is time_duration
