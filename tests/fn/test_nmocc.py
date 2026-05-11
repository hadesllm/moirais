"""Tests for morie.fn.nmocc -- OC classification rate"""

import numpy as np
import pytest

from morie.fn.nmocc import oc_classify


class TestOcClassify:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = oc_classify(data)
        assert result.value is not None

    def test_output_type(self):
        result = oc_classify(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
