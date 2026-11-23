from __future__ import annotations

from typing import Any

from ....constants import DEFAULT_SEED_MODE


def resolve_seed(seed_node: Any, seed_value: int, mode: str, even: bool, unique_id: str) -> int:
    seed, = seed_node.get_value(seed_value, mode, even, unique_id)
    return seed


def seed_change_token(mode: str) -> float | int:
    if mode == DEFAULT_SEED_MODE:
        return 0

    return float("NaN")
