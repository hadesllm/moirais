"""Tests for moirais.fn.zxait -- Aitchison compositional spatial"""

import numpy as np
import pytest

from moirais.fn.zxait import aitchison_sp


class TestAitchisonSp:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = aitchison_sp(data)
        assert result.value is not None

    def test_output_type(self):
        result = aitchison_sp(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
