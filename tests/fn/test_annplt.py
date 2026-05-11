"""Tests for morie.fn.annplt -- annotated signal plot."""

import numpy as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")

from morie.fn.annplt import annplt


class TestAnnPlt:
    def test_with_annotations(self):
        x = np.sin(np.linspace(0, 4 * np.pi, 1000))
        annotations = np.array([100, 350, 600, 850])
        result = annplt(x, fs=100, annotations=annotations)
        assert result.name == "annotated_signal"
        assert result.value == 4
        assert result.extra["n_annotations"] == 4
        plt.close(result.extra["figure"])

    def test_no_annotations(self):
        x = np.random.default_rng(42).standard_normal(500)
        result = annplt(x, fs=50)
        assert result.value == 0
        plt.close(result.extra["figure"])
