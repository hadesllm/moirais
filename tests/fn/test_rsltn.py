"""Test freq_resolution (rsltn)."""
from moirais.fn.rsltn import freq_resolution, rsltn
from moirais.fn._containers import DescriptiveResult


class TestRsltn:
    def test_basic(self):
        result = freq_resolution(1000.0, 100)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "freq_resolution"
        assert result.value == 10.0

    def test_known_value(self):
        result = freq_resolution(44100.0, 1024)
        assert abs(result.value - 44100.0 / 1024) < 1e-10

    def test_alias(self):
        assert rsltn is freq_resolution
