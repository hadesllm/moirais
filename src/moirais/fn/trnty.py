"""The whole is greater than the sum of its parts. — Aristotle"""

from __future__ import annotations

from ._containers import DescriptiveResult

_AV = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.20}
_AC = {"L": 0.77, "H": 0.44}
_PR_U = {"N": 0.85, "L": 0.62, "H": 0.27}
_PR_C = {"N": 0.85, "L": 0.68, "H": 0.50}
_UI = {"N": 0.85, "R": 0.62}
_S = {"U": False, "C": True}
_CIA = {"N": 0.0, "L": 0.22, "H": 0.56}


def cvss_base(
    *,
    av: str = "N",
    ac: str = "L",
    pr: str = "N",
    ui: str = "N",
    s: str = "U",
    c: str = "H",
    i: str = "H",
    a: str = "H",
) -> DescriptiveResult:
    """Compute CVSS v3.1 base score from metric values.

    Parameters
    ----------
    av : str
        Attack Vector: N(etwork), A(djacent), L(ocal), P(hysical).
    ac : str
        Attack Complexity: L(ow), H(igh).
    pr : str
        Privileges Required: N(one), L(ow), H(igh).
    ui : str
        User Interaction: N(one), R(equired).
    s : str
        Scope: U(nchanged), C(hanged).
    c, i, a : str
        Confidentiality / Integrity / Availability Impact: N(one), L(ow), H(igh).

    Returns
    -------
    DescriptiveResult
        ``value`` is the CVSS base score (0.0 -- 10.0).
    """
    import math

    if av not in _AV:
        raise ValueError(f"av must be one of {list(_AV)}")
    if ac not in _AC:
        raise ValueError(f"ac must be one of {list(_AC)}")
    if pr not in _PR_U:
        raise ValueError(f"pr must be one of {list(_PR_U)}")
    if ui not in _UI:
        raise ValueError(f"ui must be one of {list(_UI)}")
    if s not in _S:
        raise ValueError(f"s must be one of {list(_S)}")
    for label, val in [("c", c), ("i", i), ("a", a)]:
        if val not in _CIA:
            raise ValueError(f"{label} must be one of {list(_CIA)}")

    scope_changed = _S[s]
    pr_map = _PR_C if scope_changed else _PR_U

    iss = 1 - (1 - _CIA[c]) * (1 - _CIA[i]) * (1 - _CIA[a])
    if scope_changed:
        impact = 7.52 * (iss - 0.029) - 3.25 * (iss - 0.02) ** 15
    else:
        impact = 6.42 * iss

    exploit = 8.22 * _AV[av] * _AC[ac] * pr_map[pr] * _UI[ui]

    if impact <= 0:
        base = 0.0
    elif scope_changed:
        base = min(1.08 * (impact + exploit), 10.0)
    else:
        base = min(impact + exploit, 10.0)

    base = math.ceil(base * 10) / 10

    vector = f"CVSS:3.1/AV:{av}/AC:{ac}/PR:{pr}/UI:{ui}/S:{s}/C:{c}/I:{i}/A:{a}"
    return DescriptiveResult(
        name="CVSS v3.1 Base Score",
        value=base,
        extra={
            "vector": vector,
            "impact_subscore": round(impact, 1),
            "exploitability_subscore": round(exploit, 1),
            "scope_changed": scope_changed,
        },
    )


trnty = cvss_base


def cheatsheet() -> str:
    return "The whole is greater than the sum of its parts. — Aristotle"
