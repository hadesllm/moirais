"""Tests for moirais.fn.svdgn -- Deegan-Packel power index"""

import numpy as np
import pytest

from moirais.fn.svdgn import deegan_packel


class TestDeeganPackel:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = deegan_packel(data)
        assert result.value is not None

    def test_output_type(self):
        result = deegan_packel(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
