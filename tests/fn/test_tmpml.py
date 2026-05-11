"""Test template_match_lib (tmpml)."""
import numpy as np
from morie.fn.tmpml import template_match_lib, tmpml
from morie.fn._containers import DescriptiveResult


class TestTemplateMatchLib:
    def test_basic(self):
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(100)
        templates = [rng.standard_normal(20) for _ in range(3)]
        result = template_match_lib(signal, templates)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "template_match_lib"

    def test_self_match(self):
        signal = np.sin(np.linspace(0, 2 * np.pi, 100))
        templates = [np.ones(100), signal.copy()]
        result = template_match_lib(signal, templates)
        assert result.extra["best_template_index"] == 1

    def test_n_templates(self):
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(50)
        templates = [rng.standard_normal(10) for _ in range(5)]
        result = template_match_lib(signal, templates)
        assert result.extra["n_templates"] == 5

    def test_alias(self):
        assert tmpml is template_match_lib
