"""Pareto frontier for the accuracy-versus-overhead figure (PRD 7.4.4;
docs/TASKS.md 5B.2). Member A's Figure 4 script joins the points this returns.
"""

from __future__ import annotations

from typing import Any, Sequence


def pareto_frontier(points: Sequence[dict[str, Any]], cost: str, value: str) -> list[dict[str, Any]]:
    """The configurations no other configuration beats on both axes (cost at
    most as high and value at least as high, strictly better on one), sorted
    by rising cost. Lower cost and higher value are better."""
    ordered = sorted(points, key=lambda p: (p[cost], -p[value]))
    frontier: list[dict[str, Any]] = []
    for point in ordered:
        if not frontier or point[value] > frontier[-1][value]:
            frontier.append(point)
    return frontier
