# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Boolean expression evaluator. 'Knowledge itself is power. -- Bacon'"""
from __future__ import annotations

from ._containers import DescriptiveResult


def boolean_eval(
    expression: str,
    variables: dict[str, int],
) -> DescriptiveResult:
    """
    Evaluate a boolean expression given variable assignments.

    Supports operators: AND (&), OR (|), NOT (~), XOR (^).
    Variables are single uppercase letters.

    :param expression: Boolean expression string (e.g. "A & (B | ~C)").
    :param variables: Dict mapping variable names to 0 or 1.
    :return: DescriptiveResult with evaluation result.
    :raises ValueError: If expression contains undefined variables.

    References
    ----------
    Boole, G. (1854). *An Investigation of the Laws of Thought*.
    Walton and Maberly.
    """
    if not expression or not expression.strip():
        raise ValueError("Expression must be non-empty.")

    for var, val in variables.items():
        if val not in (0, 1):
            raise ValueError(f"Variable {var} must be 0 or 1, got {val}.")

    expr = expression
    for var in sorted(variables.keys(), key=len, reverse=True):
        expr = expr.replace(var, str(bool(variables[var])))

    expr = expr.replace("~", " not ")
    expr = expr.replace("&", " and ")
    expr = expr.replace("|", " or ")
    expr = expr.replace("^", " != ")

    allowed = set("TrueFalsendorat!=() 01")
    cleaned = expr.replace("not", "").replace("and", "").replace("or", "")
    for ch in cleaned:
        if ch not in allowed and not ch.isspace():
            raise ValueError(f"Unexpected character in expression: '{ch}'")

    result = int(bool(eval(expr)))

    return DescriptiveResult(
        name="Boolean Evaluation",
        value=result,
        extra={
            "expression": expression,
            "variables": variables,
            "result_bool": bool(result),
        },
    )


short = boolean_eval


def cheatsheet() -> str:
    return "boolean_eval({}) -> Boolean expression evaluator. 'Let the Wookiee win.' -- C-3P"
