"""Test param_count."""
from moirais.fn.prmsz import param_count, prmsz
from moirais.fn._containers import DescriptiveResult


class TestParamCount:
    def test_basic(self):
        shapes = [(768, 768), (768,), (768, 3072)]
        result = param_count(shapes)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "param_count"

    def test_total(self):
        shapes = [(10, 20), (20,)]
        result = param_count(shapes)
        assert result.value == 220

    def test_trainable_mask(self):
        shapes = [(10, 20), (20,)]
        result = param_count(shapes, trainable_mask=[True, False])
        assert result.extra["trainable"] == 200
        assert result.extra["frozen"] == 20

    def test_alias(self):
        assert prmsz is param_count
