from __future__ import annotations

import random
from datetime import datetime

from server import PromptServer

from ...constants import CATEGORY_PRIMITIVES, SEED_FEEDBACK_EVENT, SEED_MODE_FIXED, SEED_MODE_RANDOMIZE, SEED_MODE_INCREMENT, SEED_MODE_DECREMENT
from ..server.state import reset_registry
from .helpers import Input

_initial_random_state = random.getstate()
random.seed(datetime.now().timestamp())
_seed_random_state = random.getstate()
random.setstate(_initial_random_state)


def new_random_seed() -> int:
    global _seed_random_state

    random.setstate(_seed_random_state)
    seed = random.randint(0, Input.seed()[1]["max"])
    _seed_random_state = random.getstate()

    return seed


def make_even_seed(seed: int) -> int:
    return int(str(seed).translate(str.maketrans("13579", "24680")))


def next_seed(seed: int, mode: str, even: bool) -> int:
    maximum = Input.seed()[1]["max"]

    if mode == SEED_MODE_RANDOMIZE:
        seed = new_random_seed()
    elif mode == SEED_MODE_INCREMENT:
        seed = min(seed + 1, maximum)
    elif mode == SEED_MODE_DECREMENT:
        seed = max(seed - 1, 0)

    if even:
        seed = make_even_seed(seed)

    return seed


class Seed:
    def __init__(self):
        self._seed: int | None = None
        self._unique_id: str | None = None

    def __del__(self):
        if self._unique_id is not None:
            reset_registry.pop(self._unique_id, None)

    @classmethod
    def INPUT_TYPES(cls):
        return Input.groups(
            required={
                "seed_value": Input.seed(),
                "mode": Input.seed_mode(),
                "even": Input.bool(True),
            },
            hidden={"unique_id": "UNIQUE_ID"},
        )

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "get_value"
    CATEGORY = CATEGORY_PRIMITIVES

    def get_value(self, seed_value: int, mode: str, even: bool, unique_id: str):
        self._ensure_node_state(unique_id, seed_value)

        if self._consume_reset(unique_id):
            self._seed = seed_value
        elif self._seed is not None:
            self._seed = next_seed(self._seed, mode, even)

        seed_value = self._seed if self._seed is not None else seed_value
        self._send_feedback(unique_id, seed_value)

        return (seed_value,)

    def _ensure_node_state(self, unique_id: str, seed: int) -> None:
        self._unique_id = unique_id

        if unique_id:
            reset_registry.setdefault(unique_id, False)

        if self._seed is None:
            self._seed = seed

    @staticmethod
    def _consume_reset(unique_id: str) -> bool:
        if not unique_id or not reset_registry.get(unique_id):
            return False

        reset_registry[unique_id] = False
        return True

    @staticmethod
    def _send_feedback(unique_id: str, seed: int) -> None:
        PromptServer.instance.send_sync(
            SEED_FEEDBACK_EVENT,
            {"node_id": unique_id, "seed": seed},
        )

    @classmethod
    def IS_CHANGED(cls, seed_value, mode, even, unique_id):
        if mode == SEED_MODE_FIXED:
            return 0

        return float("NaN")
