# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Constraint satisfaction solver. 'No one gets through.' -- Barricade"""

from __future__ import annotations

from ._containers import DescriptiveResult


def csp_backtrack(
    variables: list[str],
    domains: dict[str, list],
    constraints: list[tuple[str, str, callable]],
) -> DescriptiveResult:
    """Solve a constraint satisfaction problem via backtracking.

    Parameters
    ----------
    variables : list of str
        Variable names.
    domains : dict
        {variable: [possible_values]}.
    constraints : list of (var1, var2, func)
        Each constraint is a tuple (v1, v2, check) where
        check(val1, val2) returns True if the assignment is valid.

    Returns
    -------
    DescriptiveResult
        With ``value`` = solution dict or None if no solution.
    """
    if not variables:
        raise ValueError("variables must be non-empty")
    for v in variables:
        if v not in domains or not domains[v]:
            raise ValueError(f"No domain for variable '{v}'")

    nodes_explored = [0]

    def is_consistent(assignment, var, val):
        for v1, v2, check in constraints:
            if v1 == var and v2 in assignment:
                if not check(val, assignment[v2]):
                    return False
            if v2 == var and v1 in assignment:
                if not check(assignment[v1], val):
                    return False
        return True

    def backtrack(assignment):
        if len(assignment) == len(variables):
            return dict(assignment)
        unassigned = [v for v in variables if v not in assignment]
        var = min(unassigned, key=lambda v: len(domains[v]))
        for val in domains[var]:
            nodes_explored[0] += 1
            if is_consistent(assignment, var, val):
                assignment[var] = val
                result = backtrack(assignment)
                if result is not None:
                    return result
                del assignment[var]
        return None

    solution = backtrack({})

    return DescriptiveResult(
        name="csp_backtrack",
        value=solution,
        extra={"n_variables": len(variables), "solved": solution is not None, "nodes_explored": nodes_explored[0]},
    )


barcd = csp_backtrack


def cheatsheet() -> str:
    return "csp_backtrack({}) -> Constraint satisfaction solver. 'No one gets through.' -- Ba"
