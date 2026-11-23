from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from nodes import CLIPTextEncode

from ...constants import *
from ..core.base import ModelConfig, PromptConfig, VideoModelConfig, clean_string, is_blank, is_not_blank
from .helpers import Input, append_info, format_info


TEXT_REPLACEMENT_DEFAULTS = {
    "find_1": "{x}",
    "find_2": "{y}",
}

PROMPT_WEIGHT_UP = 1.1
PROMPT_WEIGHT_DOWN = 0.9


@dataclass(slots=True, frozen=True)
class PromptText:
    positive: str
    negative: str


class PromptInputs:
    @staticmethod
    def text(default: str = ""):
        return Input.string(default, multiline=True)

    @staticmethod
    def line(default: str = ""):
        return Input.string(default, multiline=False)

    @staticmethod
    def optional_line():
        return Input.force("STRING")

    @staticmethod
    def replacement_required():
        return {
            "find_1": PromptInputs.line(TEXT_REPLACEMENT_DEFAULTS["find_1"]),
            "find_2": PromptInputs.line(TEXT_REPLACEMENT_DEFAULTS["find_2"]),
        }

    @staticmethod
    def replacement_optional():
        return {
            "replace_1": PromptInputs.optional_line(),
            "replace_2": PromptInputs.optional_line(),
        }


class PromptCleaner:
    def __init__(self, *, process_tag_weights: bool, find_1: str, find_2: str, replace_1: str, replace_2: str) -> None:
        self.process_tag_weights = process_tag_weights
        self.replacements = [
            (clean_string(find_1), clean_string(replace_1)),
            (clean_string(find_2), clean_string(replace_2)),
        ]

    def clean(self, text: str | None) -> str:
        result = text or ""

        for find, replace in self.replacements:
            if is_not_blank(find):
                result = result.replace(find, replace or "")

        # Cleaned after the replacements rather than before: an empty replacement leaves the commas that
        # surrounded its tag behind, which is the same debris a pasted prompt arrives with.
        result = clean_prompt_text(result)

        if self.process_tag_weights:
            result = process_tag_weights(result)

        return result


# Zero-width joiners and the BOM survive a whitespace collapse - str.split does not consider them spaces -
# and reach the encoder as a token of their own.
_PROMPT_INVISIBLE = re.compile("[\u200b-\u200d\u2060\ufeff]")
_PROMPT_COMMAS = re.compile(r"\s*,[\s,]*")


def clean_prompt_text(value: str | None) -> str:
    """One line, with the debris a prompt pasted out of a browser carries taken off it.

    Applied to each part before they are joined, never to the joined text: a `BREAK` or `;` separator would
    otherwise be cut apart by the comma rules.
    """
    text = clean_string(_PROMPT_INVISIBLE.sub("", value or "")) or ""
    # One pass covers " ,", ",,", and ", ,": a comma with anything blank or comma-shaped after it.
    return _PROMPT_COMMAS.sub(", ", text).strip().strip(",").strip()


def _join_prompt(parts: Iterable[str | None], separator: str) -> str:
    return separator.join(part for part in parts if is_not_blank(part))


def _positive_separator(separator: str) -> str:
    if separator == "break":
        return " BREAK "

    if separator == PROMPT_SEPARATOR_SEMICOLON:
        return " ; "

    return ", "


def _negative_separator(separator: str) -> str:
    return " ; " if separator == PROMPT_SEPARATOR_SEMICOLON else ", "


def _positive_parts(config: PromptConfig) -> list[str]:
    if config.positive_style_position == "start":
        return [config.positive_style, config.positive, config.posture, config.background]

    if config.positive_style_position == "end":
        return [config.positive, config.posture, config.background, config.positive_style]

    return [config.positive, config.positive_style, config.posture, config.background]


def build_prompt_text(config: PromptConfig) -> PromptText:
    return PromptText(
        positive=_join_prompt(_positive_parts(config), _positive_separator(config.separator)),
        negative=_join_prompt([config.negative_style, config.negative], _negative_separator(config.separator)),
    )


def build_region_prompt_text(config: PromptConfig | None, region_positive: str, region_negative: str) -> PromptText:
    """The prompt a detailed crop gets: the shared style plus the region text, on both sides.

    Only the two *style* parts cross over. Told "1girl, silver hair", a crop holding nothing but a hand draws
    a second girl inside that hand as soon as denoise is high enough, so the shared `positive` is left out;
    the shared `negative` is left out for the mirror reason - it is written against the scene ("crowds",
    "bad anatomy" aimed at a pose) and inside a crop it pushes the sampler away from what the frame already
    holds. `negative_style` is the general list, and that is what a crop needs.
    """
    # The parts a PromptConfig carries went through PromptCleaner on the way in; these two come straight
    # off a widget, so they get the same cleaning and the same weight shorthand here.
    weights = config.process_tag_weights if config is not None else DEFAULT_PROCESS_TAG_WEIGHTS
    positive = _cleaned_region_part(region_positive, weights)
    negative = _cleaned_region_part(region_negative, weights)

    if config is None:
        return PromptText(positive=positive, negative=negative)

    return PromptText(
        positive=_join_prompt([config.positive_style, positive], _positive_separator(config.separator)),
        negative=_join_prompt([config.negative_style, negative], _negative_separator(config.separator)),
    )


def _cleaned_region_part(text: str, process_weights: bool) -> str:
    cleaned = clean_prompt_text(text)
    return process_tag_weights(cleaned) if process_weights else cleaned


@dataclass(slots=True, frozen=True)
class RegionSections:
    """A region widget split into the text every region gets and the text addressed to one of them."""
    common: str
    sections: dict[int, str]

    def for_region(self, number: int) -> str:
        return _join_prompt([self.common, self.sections.get(number)], ", ")

    def skips_region(self, number: int) -> bool:
        """`2: -` - a section holding nothing but the marker means that region is not wanted at all.

        Read off the raw section, before the text is cleaned and joined to the shared parts: by the time
        it is a prompt a lone dash is indistinguishable from an empty one.
        """
        return (self.sections.get(number) or "").strip() == DETAILER_REGION_SKIP


# `1: tags` opens a section that runs to the next such line. Whatever stands before the first one is shared,
# which is what a widget with no numbers in it at all consists of.
_REGION_SECTION = re.compile(r"^\s*(\d+)\s*:\s*(.*)$")


def parse_region_sections(text: str | None) -> RegionSections:
    common: list[str] = []
    sections: dict[int, list[str]] = {}
    current: list[str] | None = None

    for line in (text or "").splitlines():
        match = _REGION_SECTION.match(line)
        if match is not None:
            # A number typed twice extends its section rather than replacing it.
            current = sections.setdefault(int(match.group(1)), [])
            line = match.group(2)

        target = current if current is not None else common
        if is_not_blank(line):
            target.append(line.strip())

    return RegionSections(
        common=_join_prompt(common, ", "),
        sections={number: _join_prompt(lines, ", ") for number, lines in sections.items() if lines},
    )


def encode_prompt(clip, config: PromptConfig):
    text = build_prompt_text(config)
    encoder = CLIPTextEncode()
    positive, = encoder.encode(clip, text.positive)
    negative, = encoder.encode(clip, text.negative)
    return positive, negative, text


def process_tag_weights(text: str | None) -> str:
    if is_blank(text):
        return ""

    def replace_weight(match: re.Match[str], factor: float, open_char: str) -> str:
        tags = match.group(1)

        if ":" in tags:
            return match.group(0)

        level = match.group(0).count(open_char)
        weight = round(factor ** level, 4)
        separator = ", " if "," in tags else " + "
        return separator.join(f"({tag.strip()}:{weight})" for tag in re.split(r"[,\\+]+", tags) if tag.strip())

    text = re.sub(r"\[+([^\[\]]+)\]+", lambda match: replace_weight(match, PROMPT_WEIGHT_DOWN, "["), text)
    text = re.sub(r"\(+([^()]+)\)+", lambda match: replace_weight(match, PROMPT_WEIGHT_UP, "("), text)
    return text


def prompt_debug_text(config: PromptConfig) -> str:
    sections = {
        "Positive": config.positive,
        "Posture": config.posture,
        "Background": config.background,
        "Negative": config.negative,
        "Positive Style": config.positive_style,
        "Negative Style": config.negative_style,
    }

    info = "\n\n".join(f"[{title}]\n{text}" for title, text in sections.items() if is_not_blank(text))
    return append_info(
        info,
        format_info(
            "Prompt Config",
            {
                "ProcessWeights": config.process_tag_weights,
                "PromptSeparator": config.separator,
                "PositiveStylePosition": config.positive_style_position,
            },
        ),
    )


class PromptConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.groups(
            required={
                "process_tag_weights": Input.bool(DEFAULT_PROCESS_TAG_WEIGHTS),
                "separator": Input.choice(PROMPT_SEPARATORS, DEFAULT_PROMPT_SEPARATOR),
                "positive_style_position": Input.choice(POSITIVE_STYLE_POSITIONS, DEFAULT_POSITIVE_STYLE_POSITION),
                **PromptInputs.replacement_required(),
                "positive": PromptInputs.text(DEFAULT_POSITIVE_PROMPT),
                "posture": PromptInputs.text(),
                "background": PromptInputs.text(),
                "negative": PromptInputs.text(DEFAULT_NEGATIVE_PROMPT),
                "positive_style": PromptInputs.text(DEFAULT_POSITIVE_STYLE),
                "negative_style": PromptInputs.text(DEFAULT_NEGATIVE_STYLE),
            },
            optional=PromptInputs.replacement_optional(),
        )

    RETURN_TYPES = ("PROMPT_CONFIG", "STRING")
    RETURN_NAMES = ("config", "prompt")
    FUNCTION = "get_value"
    CATEGORY = CATEGORY_PROMPT_CONFIG

    def get_value(
        self,
        process_tag_weights: bool,
        separator: str,
        positive_style_position: str,
        find_1: str,
        find_2: str,
        positive: str,
        posture: str,
        background: str,
        negative: str,
        positive_style: str,
        negative_style: str,
        replace_1: str = "",
        replace_2: str = "",
    ):
        cleaner = PromptCleaner(
            process_tag_weights=process_tag_weights,
            find_1=find_1,
            find_2=find_2,
            replace_1=replace_1,
            replace_2=replace_2,
        )
        config = PromptConfig(
            positive=cleaner.clean(positive),
            posture=cleaner.clean(posture),
            background=cleaner.clean(background),
            negative=cleaner.clean(negative),
            positive_style=cleaner.clean(positive_style),
            negative_style=cleaner.clean(negative_style),
            separator=separator,
            positive_style_position=positive_style_position,
            process_tag_weights=process_tag_weights,
        )

        return config, prompt_debug_text(config)


class ModelConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.groups(
            required={"model": ("MODEL",), "clip": ("CLIP",), "config": ("PROMPT_CONFIG",)},
            # The names belong to the model they describe, so they travel with it.
            optional={
                "model_name": Input.force("STRING"),
                "loras_name": Input.force("STRING"),
            },
        )

    RETURN_TYPES = ("MODEL_CONFIG", "STRING", "STRING")
    RETURN_NAMES = ("model_config", "positive_text", "negative_text")
    FUNCTION = "get_value"
    CATEGORY = CATEGORY_PROMPT_CONFIG

    def get_value(self, model, clip, config: PromptConfig, model_name: str | None = None, loras_name: str | None = None):
        positive, negative, text = encode_prompt(clip, config)
        model_config = ModelConfig(
            model=model,
            clip=clip,
            positive=positive,
            negative=negative,
            positive_text=text.positive,
            negative_text=text.negative,
            prompt=config.get_copy(),
            model_name=model_name or "",
            loras_name=loras_name or "",
        )
        return model_config, text.positive, text.negative


class VideoModelConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.groups(
            required={
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                # Next to model_low because it only means anything once there is a second expert to hand over to.
                "boundary": Input.wan_boundary(DEFAULT_WAN_BOUNDARY),
                # A video prompt is a sentence or two describing motion, not a wall of tags, so it is typed
                # here rather than assembled by a node of its own.
                "positive": PromptInputs.text(DEFAULT_VIDEO_POSITIVE_PROMPT),
                "negative": PromptInputs.text(),
            },
            optional={
                # The Wan 2.2 low-noise expert, and the encoder only Wan 2.1 I2V can project.
                "model_low": ("MODEL",),
                "clip_vision": ("CLIP_VISION",),
                "model_name": Input.force("STRING"),
                "loras_name": Input.force("STRING"),
            },
        )

    RETURN_TYPES = ("VIDEO_MODEL_CONFIG", "STRING", "STRING")
    RETURN_NAMES = ("video_model_config", "positive_text", "negative_text")
    FUNCTION = "get_value"
    CATEGORY = CATEGORY_PROMPT_CONFIG

    def get_value(self, model, clip, vae, boundary: float, positive: str, negative: str, model_low=None,
                  clip_vision=None, model_name: str | None = None, loras_name: str | None = None):
        # Video prompts are plain sentences, so the ((word)) weight shorthand is off: cleaning is all they need.
        cleaner = PromptCleaner(process_tag_weights=False, find_1="", find_2="", replace_1="", replace_2="")
        config = PromptConfig(
            positive=cleaner.clean(positive),
            negative=cleaner.clean(negative),
            positive_style="",
            negative_style="",
            process_tag_weights=False,
        )
        positive, negative, text = encode_prompt(clip, config)
        video_model_config = VideoModelConfig(
            model=model,
            model_low=model_low,
            clip=clip,
            vae=vae,
            clip_vision=clip_vision,
            positive=positive,
            negative=negative,
            positive_text=text.positive,
            negative_text=text.negative,
            model_name=model_name or "",
            loras_name=loras_name or "",
            boundary=boundary,
        )
        return video_model_config, text.positive, text.negative
