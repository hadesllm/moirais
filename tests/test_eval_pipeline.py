"""Auto-discovers + runs every DatasetGate in
moirais.eval_pipeline.SEED_GATES (plus anything downstream code
has registered).

Behaviour:
    pass  -> assertion succeeds.
    skip  -> pytest.skip (remote endpoint unreachable in the env,
             e.g. offline CI — treated as inconclusive, not red).
    fail  -> hard fail, with the per-key failure list.
    error -> hard fail, with the exception type + message.

Strict mode is governed by env var ``MOIRAIS_PIPELINE_STRICT=1`` —
when set, "skip" also becomes a hard failure (so an offline CI run
is flagged rather than silently treated as green). Off by default
because most contributors won't have a configured remote endpoint
or the bundled local SQLite mirrors.
"""
from __future__ import annotations

import os

import pytest

from moirais import eval_pipeline as ep


_GATES = ep.gates()
STRICT = os.getenv("MOIRAIS_PIPELINE_STRICT", "0") == "1"


@pytest.mark.skipif(not _GATES, reason="no DatasetGates registered")
@pytest.mark.parametrize("gate", _GATES, ids=[g.name for g in _GATES])
def test_dataset_gate(gate: ep.DatasetGate) -> None:
    result = ep.run_gate(gate)
    if result.status == "pass":
        return
    if result.status == "skip":
        if STRICT:
            pytest.fail(f"{gate.name}: skipped — {result.detail} (STRICT mode)")
        pytest.skip(f"{gate.name}: {result.detail}")
    if result.status == "error":
        pytest.fail(f"{gate.name}: errored — {result.detail}")
    # fail
    pytest.fail(
        f"{gate.name}: {len(result.failures)} expectation(s) violated\n"
        + "\n".join(f"  {f}" for f in result.failures)
        + f"\n  actual: {result.actual!r}\n"
        + f"  citation: {gate.citation}"
    )


def test_module_imports() -> None:
    assert callable(ep.run_all)
    assert callable(ep.summary)
    assert callable(ep.register)
    assert isinstance(ep.SEED_GATES, list)


def test_summary_shape() -> None:
    s = ep.summary(ep.run_all())
    assert set(s.keys()) >= {"pass", "fail", "skip", "error", "total"}
    assert s["total"] == len(_GATES)
