"""Test template_match_detect."""
import numpy as np
from morie.fn.tmplm import template_match_detect, alias
from morie.fn._containers import DescriptiveResult


class TestTemplateMatchDetect:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = template_match_detect(x, template=x[50:70])
        assert isinstance(result, DescriptiveResult)

    def test_value(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = template_match_detect(x, template=x[50:70])
        assert isinstance(result.value, int)
        assert result.value >= 0

    def test_extra_keys(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = template_match_detect(x, template=x[50:70])
        assert "indices" in result.extra
        assert "correlations" in result.extra

    def test_name(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = template_match_detect(x, template=x[50:70])
        assert result.name == "template_match"

    def test_alias(self):
        assert alias is template_match_detect
