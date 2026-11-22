"""Helpers for summarizing and recording benchmark measurements."""

from __future__ import annotations

import math
from collections.abc import Iterable


def percentile(values: Iterable[float], percentile: float) -> float:
    """Return a linear-interpolated percentile for a non-empty value list."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("values must not be empty")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")

    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)