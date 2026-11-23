from __future__ import annotations

import os
from collections import Counter
from typing import Any

import folder_paths

from ...constants import EXPERT_PAIR_SUFFIX, FOLDER_UPSCALE_MODELS

# Loaded upscalers by widget name, with the (file, mtime) they were read from.
_upscale_models: dict[str, tuple[tuple[str, float], Any]] = {}


def model_names(folder: str) -> list[str]:
    return [os.path.splitext(path)[0] for path in folder_paths.get_filename_list(folder)]


# Wan 2.2 ships its models and LoRAs as a high-noise/low-noise pair whose names differ in that one word.
_EXPERT_TOKENS = (("high", "low"), ("HIGH", "LOW"), ("High", "Low"))


def expert_names(folder: str, pairs_only: bool = False) -> list[str]:
    """What the chooser offers: one entry per pair, named up to its high/low word, unpaired ones after."""
    pairs, unpaired = _expert_entries(folder)
    names = [display for display, _high, _low in pairs]

    return names if pairs_only else names + unpaired


def expert_pair(name: str, folder: str) -> tuple[str, str]:
    """The (high, low) files behind a chosen entry — the same name twice when it stands for one file."""
    pairs, _unpaired = _expert_entries(folder)
    for display, high, low in pairs:
        if name == display:
            return high, low

    # Not a folded entry, so it names a file directly: an unpaired one, or a value saved by a graph built
    # before the list started folding pairs.
    return _expert_twin(name, _known_names(folder))


def _expert_entries(folder: str) -> tuple[list[tuple[str, str, str]], list[str]]:
    known = _known_names(folder)
    pairs: list[tuple[str, str, str]] = []
    unpaired: list[str] = []

    for name in model_names(folder):
        high, low = _expert_twin(name, known)
        if high == low:
            unpaired.append(name)
        elif name == high:
            # The low half is never listed: its twin already stands for the whole pair.
            pairs.append((_pair_display_name(high), high, low))

    return _disambiguated(pairs), unpaired


def _pair_display_name(high: str) -> str:
    for high_token, _low_token in _EXPERT_TOKENS:
        index = high.find(high_token)
        # An empty stem means the whole name is the word, and there would be nothing left to read.
        if index > 0:
            return f"{high[:index].rstrip('-_. ')}{EXPERT_PAIR_SUFFIX}"

    return f"{high}{EXPERT_PAIR_SUFFIX}"


def _disambiguated(pairs: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    # Cutting at the word collapses "..._high_noise" and "..._high_noise_fp8" onto one string, and a combo
    # with the same entry twice cannot be chosen from. Those fall back to the full name of their high half.
    counts = Counter(display for display, _high, _low in pairs)

    return [
        (display if counts[display] == 1 else f"{high}{EXPERT_PAIR_SUFFIX}", high, low)
        for display, high, low in pairs
    ]


def _expert_twin(name: str, known) -> tuple[str, str]:
    for high_token, low_token in _EXPERT_TOKENS:
        if high_token in name:
            twin = name.replace(high_token, low_token)
            if known(twin):
                return name, twin

        if low_token in name:
            twin = name.replace(low_token, high_token)
            if known(twin):
                return twin, name

    return name, name


def _known_names(folder: str):
    # The swap is confirmed against the folder rather than against a word boundary: "slow" and "highlight"
    # produce a candidate that simply does not exist, and an unpaired model then goes to both experts.
    names = set(model_names(folder))
    # The plain name first: splitext alone would cut "wan2.2_low_noise" down to "wan2", because the last
    # dot in a Wan name belongs to the version, not to an extension. The second form is for a saved value
    # that kept its ".safetensors".
    return lambda candidate: candidate in names or os.path.splitext(candidate)[0] in names


def load_upscale_model(model_name: str) -> Any:
    """The upscaler, read from disk once per file version and shared by every run after that.

    A config is rebuilt on every execution, so a cache on the config lived for one run and the weights were
    read again each time. Keyed by the modification time too, so a replaced file is picked up.
    """
    # Imported here: comfy_extras pulls in torch, and constants-only importers of this module should not pay for it.
    from comfy_extras.nodes_upscale_model import UpscaleModelLoader

    from .base import get_path_by_filename

    name = get_path_by_filename(model_name, FOLDER_UPSCALE_MODELS)
    key = (name, os.path.getmtime(folder_paths.get_full_path_or_raise(FOLDER_UPSCALE_MODELS, name)))

    if _upscale_models.get(model_name, (None, None))[0] != key:
        upscale_model, = UpscaleModelLoader().load_model(name)
        _upscale_models[model_name] = (key, upscale_model)

    return _upscale_models[model_name][1]
