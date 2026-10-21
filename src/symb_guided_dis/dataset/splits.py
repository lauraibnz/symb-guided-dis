"""Shared, group-aware train/validation/test split utilities."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping


SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_SPLIT_FRACTIONS = {
    "train": 0.80,
    "validation": 0.10,
    "test": 0.10,
}
DEFAULT_SPLIT_SEED = 42


def normalize_split_fractions(
    fractions: Mapping[str, float] | Iterable[float] | str | None = None,
) -> dict[str, float]:
    """Validate and normalize a train/validation/test fraction specification.

    Strings use the compact CLI form ``"0.8,0.1,0.1"``.  Parser code receives
    the normalized dictionary, which keeps dataset-specific parsers independent
    from argparse.
    """
    if fractions is None:
        result = dict(DEFAULT_SPLIT_FRACTIONS)
    elif isinstance(fractions, str):
        values = [part.strip() for part in fractions.split(",") if part.strip()]
        if len(values) != len(SPLIT_NAMES):
            raise ValueError(
                "Split fractions must be three comma-separated values for "
                f"{SPLIT_NAMES}, got {fractions!r}."
            )
        result = {name: float(value) for name, value in zip(SPLIT_NAMES, values)}
    elif isinstance(fractions, Mapping):
        result = {name: float(fractions[name]) for name in SPLIT_NAMES if name in fractions}
    else:
        values = list(fractions)
        if len(values) != len(SPLIT_NAMES):
            raise ValueError(
                f"Split fractions must contain {len(SPLIT_NAMES)} values, got {len(values)}."
            )
        result = {name: float(value) for name, value in zip(SPLIT_NAMES, values)}

    if set(result) != set(SPLIT_NAMES):
        raise ValueError(
            "Split fractions must provide exactly train, validation, and test; "
            f"got {sorted(result)}."
        )
    if any(value <= 0.0 for value in result.values()):
        raise ValueError(f"Split fractions must all be positive, got {result}.")
    if abs(sum(result.values()) - 1.0) > 1e-8:
        raise ValueError(f"Split fractions must sum to 1, got {result}.")
    return result


def assign_group_splits(
    group_ids: Iterable[str],
    *,
    fractions: Mapping[str, float] | Iterable[float] | str | None = None,
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, str]:
    """Assign every unique logical piece/group to one deterministic split."""
    fractions = normalize_split_fractions(fractions)
    groups = sorted({str(group_id) for group_id in group_ids})
    if len(groups) < len(SPLIT_NAMES):
        raise ValueError(
            f"Need at least {len(SPLIT_NAMES)} split groups, got {len(groups)}."
        )

    shuffled_groups = list(groups)
    random.Random(int(seed)).shuffle(shuffled_groups)
    validation_count = max(1, round(len(groups) * fractions["validation"]))
    test_count = max(1, round(len(groups) * fractions["test"]))
    if validation_count + test_count >= len(groups):
        raise ValueError(
            f"Split allocation leaves no train groups: groups={len(groups)}, fractions={fractions}."
        )

    assignments = {group_id: "validation" for group_id in shuffled_groups[:validation_count]}
    assignments.update(
        {group_id: "test" for group_id in shuffled_groups[validation_count : validation_count + test_count]}
    )
    assignments.update(
        {group_id: "train" for group_id in shuffled_groups[validation_count + test_count :]}
    )
    return assignments
