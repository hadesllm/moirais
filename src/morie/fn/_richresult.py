"""Rich-output result containers for morie.fn.

R-style verbose ``__repr__`` so users see paragraph-level summaries
when they print a result, not just a bare dict -- modeled after
psych::omega(), lavaan summary(), survival::summary.coxph(), etc.

Usage::

    res = welcht([1,2,3], [4,5,6])
    print(res)        # multi-section ASCII summary
    res.statistic     # still attribute-accessible like a dict
    res.warnings      # list of advisory strings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RichResult(dict):
    """Generic rich-printing result dict-like -- psych::omega style.

    Inherits from `dict` so `isinstance(result, dict)` is True and
    legacy callers that expect a plain dict (`.get(...)`, `for k in r`,
    `result["statistic"]`) keep working unchanged. The RichResult
    layer adds title / sections / tables / interpretation / warnings
    on top -- opt-in via `print(result)` or `result.summary()`.

    Supports multi-section output:
    - title              short test/model name (top of output)
    - call               optional call signature line ("Call: foo(x, y, …)")
    - summary_lines      headline metrics: list of (label, value) tuples
    - sections           list of {"title": str, "lines": [(lbl, val)], "table": [...]} dicts
                         for additional sections (each printed with its own header)
    - tables             list of {"title": str, "headers": [...], "rows": [[...]]}
                         for tabular output (column-aligned)
    - comparison         optional dict with same shape as a section,
                         printed under "Compare this with …" heading
    - extras             list of free-form paragraph blocks
    - warnings           list of advisory strings -- surfaced VERBATIM,
                         multi-line OK
    - interpretation     optional plain-language concluding sentence(s)
    - payload            arbitrary dict for programmatic access
    """

    title: str = ""
    call: str = ""
    summary_lines: list[tuple[str, Any]] = field(default_factory=list)
    sections: list[dict] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)
    comparison: dict | None = None
    extras: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    interpretation: str = ""
    payload: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Populate the dict portion (self) from payload so
        # isinstance(self, dict) is True and dict.* methods work.
        # We don't store payload twice -- `payload` and `self` (as a dict)
        # are kept in sync because __getitem__/__contains__/keys/values
        # all proxy through payload.
        try:
            dict.update(self, self.payload)
        except Exception:
            pass

    # Allow attribute access: result.statistic instead of result.payload["statistic"]
    def __getattr__(self, name: str) -> Any:
        if name == "payload":
            raise AttributeError(name)
        try:
            return self.payload[name]
        except KeyError:
            raise AttributeError(
                f"{type(self).__name__!r} has no attribute {name!r}"
            ) from None

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def get(self, key: str, default=None):
        return self.payload.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.payload

    def keys(self):
        return self.payload.keys()

    def values(self):
        return self.payload.values()

    def items(self):
        return self.payload.items()

    def to_dict(self) -> dict:
        return dict(self.payload)

    # ── Rendering ──────────────────────────────────────────────────

    def _render(self) -> str:
        out: list[str] = []
        # Surface warnings BEFORE the result block -- psych::omega does this,
        # because users want to see "Matrix not positive definite" up front,
        # not buried at the bottom.
        if self.warnings:
            for w in self.warnings:
                out.append(f"Warning: {w}")
            out.append("")
        if self.title:
            out.append(self.title)
            out.append("=" * len(self.title))
        if self.call:
            out.append(f"Call: {self.call}")
        if self.title or self.call:
            out.append("")
        if self.summary_lines:
            label_w = max((len(str(L)) for L, _ in self.summary_lines), default=8)
            for label, value in self.summary_lines:
                out.append(f"  {label:<{label_w}}  {self._fmt(value)}")
            out.append("")
        # Sections (each with its own header + content)
        for sec in self.sections:
            sec_title = sec.get("title", "")
            if sec_title:
                out.append(sec_title)
                out.append("-" * len(sec_title))
            for label, value in sec.get("lines", []):
                out.append(f"  {label}  {self._fmt(value)}")
            if "table" in sec and sec["table"]:
                out.append(self._render_table(sec.get("headers", []), sec["table"]))
            if "text" in sec and sec["text"]:
                out.append(sec["text"])
            out.append("")
        # Stand-alone tables
        for tbl in self.tables:
            tbl_title = tbl.get("title", "")
            if tbl_title:
                out.append(tbl_title)
            out.append(self._render_table(tbl.get("headers", []), tbl.get("rows", [])))
            out.append("")
        # Comparison block (psych::omega's "Compare this with…")
        if self.comparison:
            cmp_t = self.comparison.get("title", "Comparison")
            out.append(f"Compare this with {cmp_t}:")
            for label, value in self.comparison.get("lines", []):
                out.append(f"  {label}  {self._fmt(value)}")
            out.append("")
        for block in self.extras:
            out.append(block)
            out.append("")
        if self.interpretation:
            out.append(self.interpretation)
        return "\n".join(out).rstrip()

    @staticmethod
    def _render_table(headers: list[str], rows: list[list[Any]]) -> str:
        """Render a column-aligned ASCII table."""
        if not rows:
            return ""
        all_rows = ([headers] if headers else []) + [
            [RichResult._fmt(c) for c in r] for r in rows
        ]
        widths = [
            max(len(str(row[i])) if i < len(row) else 0 for row in all_rows)
            for i in range(max(len(r) for r in all_rows))
        ]
        lines = []
        for r_idx, r in enumerate(all_rows):
            cells = [
                f"{str(c):<{widths[i]}}"
                for i, c in enumerate(r)
            ]
            lines.append("  " + "  ".join(cells))
            if r_idx == 0 and headers:
                lines.append("  " + "  ".join("-" * w for w in widths))
        return "\n".join(lines)

    @staticmethod
    def _fmt(v: Any) -> str:
        if isinstance(v, float):
            if abs(v) < 1e-4 and v != 0:
                return f"{v:.3e}"
            return f"{v:.4f}".rstrip("0").rstrip(".")
        if isinstance(v, (list, tuple)):
            inner = ", ".join(RichResult._fmt(x) for x in v[:6])
            if len(v) > 6:
                inner += ", …"
            return f"[{inner}]"
        return str(v)

    def __repr__(self) -> str:
        return self._render()

    __str__ = __repr__

    # Convenience: float(result) returns the headline scalar so callers
    # that historically did `d = cohend(x, y)` still work after upgrade.
    # Looks for payload["value"], payload["statistic"], payload["score"] --
    # in that priority order.
    def __float__(self) -> float:
        for key in ("value", "statistic", "score", "estimate"):
            v = self.payload.get(key)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        raise TypeError(
            f"{type(self).__name__} has no scalar headline "
            f"(none of {list(self.payload.keys())[:8]})"
        )

    # Scalar comparison operators -- let `cohend(x, y) == 1.0` and
    # `cohend(x, y) < 0.5` and `pytest.approx(0.25) == mcfadr(...)`
    # work without users having to wrap in float() first.
    #
    # __eq__ also accepts another RichResult or anything with a
    # __float__ -- falls back to identity comparison if no scalar exists.
    def _try_float(self) -> float | None:
        try:
            return float(self)
        except TypeError:
            return None

    def __eq__(self, other) -> bool:
        # If `other` is a RichResult too, compare headlines.
        if isinstance(other, RichResult):
            sv = self._try_float()
            ov = other._try_float()
            if sv is None or ov is None:
                return self is other
            return sv == ov
        # If `other` is numeric (int/float/numpy/pytest.approx), compare
        # the headline scalar.
        sv = self._try_float()
        if sv is None:
            return NotImplemented
        try:
            return sv == other
        except Exception:  # pragma: no cover
            return NotImplemented

    def __ne__(self, other) -> bool:
        eq = self.__eq__(other)
        if eq is NotImplemented:
            return NotImplemented
        return not eq

    def __lt__(self, other) -> bool:
        sv = self._try_float()
        if sv is None:
            return NotImplemented
        return sv < (float(other) if isinstance(other, RichResult) else other)

    def __le__(self, other) -> bool:
        sv = self._try_float()
        if sv is None:
            return NotImplemented
        return sv <= (float(other) if isinstance(other, RichResult) else other)

    def __gt__(self, other) -> bool:
        sv = self._try_float()
        if sv is None:
            return NotImplemented
        return sv > (float(other) if isinstance(other, RichResult) else other)

    def __ge__(self, other) -> bool:
        sv = self._try_float()
        if sv is None:
            return NotImplemented
        return sv >= (float(other) if isinstance(other, RichResult) else other)

    # __hash__ must be re-declared whenever __eq__ is overridden, or
    # instances become unhashable. Use id() -- RichResults aren't intended
    # as dict keys, so identity-hash is fine.
    def __hash__(self) -> int:
        return id(self)

    # Unary numeric protocol -- let abs(result), -result, +result work
    def __abs__(self) -> float:
        return abs(float(self))

    def __neg__(self) -> float:
        return -float(self)

    def __pos__(self) -> float:
        return +float(self)

    # Binary arithmetic with scalars / other RichResults -- operates on
    # the headline scalar. Returns a plain float; chain operations
    # don't preserve the structured side of the result, which is correct
    # (the structured fields belong to the original computation).
    def _other_as_float(self, other) -> float:
        if isinstance(other, RichResult):
            return float(other)
        return float(other)

    def __add__(self, other) -> float:
        return float(self) + self._other_as_float(other)

    def __radd__(self, other) -> float:
        return self._other_as_float(other) + float(self)

    def __sub__(self, other) -> float:
        return float(self) - self._other_as_float(other)

    def __rsub__(self, other) -> float:
        return self._other_as_float(other) - float(self)

    def __mul__(self, other) -> float:
        return float(self) * self._other_as_float(other)

    def __rmul__(self, other) -> float:
        return self._other_as_float(other) * float(self)

    def __truediv__(self, other) -> float:
        return float(self) / self._other_as_float(other)

    def __rtruediv__(self, other) -> float:
        return self._other_as_float(other) / float(self)

    # Boolean cast -- important for `if result:` checks (default truthy)
    def __bool__(self) -> bool:
        try:
            return bool(float(self))
        except TypeError:
            return True  # has no scalar but exists, so truthy

    # round(result, n)
    def __round__(self, ndigits: int = 0) -> float:
        return round(float(self), ndigits)

    # Convenience: pythonic .summary() returning the text
    def summary(self) -> str:
        return self._render()


def hypothesis_test_result(
    *,
    test_name: str,
    statistic: float,
    pvalue: float,
    df: float | None = None,
    alpha: float = 0.05,
    extra_summary: list[tuple[str, Any]] | None = None,
    warnings: list[str] | None = None,
    extra_payload: dict | None = None,
    callable_name: str | None = None,
) -> RichResult:
    """Factory for hypothesis-test results with a uniform layout.

    `callable_name` (optional) -- short name of the morie.fn callable
    that produced this result. If provided, a "Learn more" pointer is
    appended to the extras suggesting `describe(<name>)` for the full
    pedagogical guide. Caller can pass it explicitly OR auto-detect
    via the inspect-stack fallback in attach_describe_pointer().
    """
    summary: list[tuple[str, Any]] = [
        ("Test statistic", statistic),
        ("p-value", pvalue),
    ]
    if df is not None:
        summary.insert(1, ("Degrees of freedom", df))
    if extra_summary:
        summary.extend(extra_summary)

    if pvalue < alpha:
        interp = (f"Reject H₀ at α={alpha}: result is statistically "
                  f"significant (p={pvalue:.4f}).")
    else:
        interp = (f"Fail to reject H₀ at α={alpha}: insufficient evidence "
                  f"(p={pvalue:.4f}).")

    payload = {"statistic": statistic, "pvalue": pvalue}
    if df is not None:
        payload["df"] = df
    if extra_payload:
        payload.update(extra_payload)

    extras = []
    # Auto-detect calling function name if not supplied
    if callable_name is None:
        callable_name = _detect_caller_name()
    if callable_name:
        extras.append(_describe_pointer(callable_name))
        payload["_callable_name"] = callable_name

    return RichResult(
        title=test_name,
        summary_lines=summary,
        extras=extras,
        warnings=warnings or [],
        interpretation=interp,
        payload=payload,
    )


def _detect_caller_name() -> str | None:
    """Walk back the call stack to find the morie.fn callable that
    invoked this factory. Returns the function name or None.
    """
    import inspect
    try:
        frame = inspect.currentframe()
        # currentframe (this fn) -> hypothesis_test_result -> caller fn.
        # Walk back exactly 2 frames so we land in the caller's frame.
        for _ in range(2):
            if frame is None:
                return None
            frame = frame.f_back
        if frame is None:
            return None
        # If the caller is in morie/fn/<name>.py, fn name should be
        # the same as the module file name.
        mod = inspect.getmodule(frame)
        if mod is None:
            return None
        modname = mod.__name__
        if "morie.fn." not in modname:
            return None
        # Strip "morie.fn." prefix
        short = modname.split("morie.fn.")[-1]
        # Skip private helpers
        if short.startswith("_"):
            return None
        return short
    except Exception:  # noqa: BLE001
        return None


def _describe_pointer(callable_name: str) -> str:
    """The 'Learn more' pointer line appended to results."""
    return (f"Learn more: `from morie.fn import describe; "
            f"print(describe({callable_name!r}))` for the full "
            f"pedagogical guide (when, why, formula, common mistakes).")


def with_describe_pointer(result: "RichResult", callable_name: str) -> "RichResult":
    """Add a describe() pointer to an existing RichResult.

    Used by callables that return RichResult directly (without going
    through hypothesis_test_result()) to opt-in to the same convention.
    """
    pointer = _describe_pointer(callable_name)
    if pointer not in (result.extras or []):
        result.extras = list(result.extras or []) + [pointer]
    result.payload["_callable_name"] = callable_name
    return result
