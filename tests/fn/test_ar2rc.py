"""Test ar_to_reflection (ar2rc)."""
import numpy as np
from morie.fn.ar2rc import ar_to_reflection, ar2rc
from morie.fn._containers import DescriptiveResult


class TestAr2rc:
    def test_basic(self):
        result = ar_to_reflection([1.0, -0.5, 0.2])
        assert isinstance(result, DescriptiveResult)
        assert result.name == "ar_to_reflection"
        rc = result.extra["rc"]
        assert len(rc) == 2

    def test_alias(self):
        assert ar2rc is ar_to_reflection
