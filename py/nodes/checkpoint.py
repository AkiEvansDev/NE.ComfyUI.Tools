from __future__ import annotations

from nodes import CheckpointLoaderSimple, CLIPLoader, VAELoader

from ...constants import *
from ..core.base import extract_filename, get_path_by_filename
from ..core.models import expert_names, expert_pair, model_names
from ..core.video_utils import load_clip_vision
from .helpers import Input, format_info


class CustomCheckpointLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(
            checkpoint=Input.choice(model_names(FOLDER_CHECKPOINTS)),
        )

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "name")
    FUNCTION = "load_checkpoint"
    CATEGORY = CATEGORY_CHECKPOINT

    def load_checkpoint(self, checkpoint: str):
        model, clip, vae = _load_checkpoint(checkpoint)
        return (model, clip, vae, extract_filename(checkpoint))


class WanCheckpointLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(
            checkpoint=Input.choice(model_names(FOLDER_CHECKPOINTS)),
            clip_vision=Input.choice(model_names(FOLDER_CLIP_VISION)),
        )

    RETURN_TYPES = ("MODEL", "CLIP_VISION", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip_vision", "clip", "vae", "model_names")
    FUNCTION = "load_checkpoint"
    CATEGORY = CATEGORY_CHECKPOINT

    def load_checkpoint(self, checkpoint: str, clip_vision: str):
        clip_vision_path = get_path_by_filename(clip_vision, FOLDER_CLIP_VISION)
        if clip_vision_path is None:
            raise ValueError(f"CLIP Vision model not found: {clip_vision}")

        model, clip, vae = _load_checkpoint(checkpoint)
        clip_vision_model = load_clip_vision(clip_vision_path)
        checkpoint_name = extract_filename(checkpoint)
        clip_vision_name = extract_filename(clip_vision)
        info = format_info(
            "Wan Loader",
            {
                "Checkpoint": checkpoint_name,
                "ClipVision": clip_vision_name,
            },
        )
        return (model, clip_vision_model, clip, vae, info)


class WanExpertCheckpointLoader:
    @classmethod
    def INPUT_TYPES(cls):
        # Pairs only: a checkpoint without a twin is a single-model run, which is what WanCheckpointLoader
        # already is. One entry stands for two files and says so.
        #
        # The text encoder and the vae are picked here rather than wired in, because a split Wan 2.2 pair
        # never carries either: the files hold nothing but `model.diffusion_model.*`. Everything the step
        # needs is chosen in one place, and all three names travel out together.
        return Input.required(
            checkpoint=Input.choice(expert_names(FOLDER_CHECKPOINTS, pairs_only=True)),
            text_encoder=Input.choice(model_names(FOLDER_TEXT_ENCODERS)),
            vae=Input.choice(model_names(FOLDER_VAE)),
        )

    RETURN_TYPES = ("MODEL", "MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model_high", "model_low", "clip", "vae", "model_names")
    FUNCTION = "load_checkpoint"
    CATEGORY = CATEGORY_CHECKPOINT

    def load_checkpoint(self, checkpoint: str, text_encoder: str, vae: str):
        high, low = expert_pair(checkpoint, FOLDER_CHECKPOINTS)

        model_high, _, _ = _load_checkpoint(high)
        # The low half is loaded for its model alone: both experts share the encoder and the vae, and a
        # second copy of either would hold a second set of weights for nothing.
        model_low, _, _ = _load_checkpoint(low) if low != high else (model_high, None, None)

        clip = _load_text_encoder(text_encoder)
        loaded_vae = _load_vae(vae)

        info = format_info(
            "Wan Expert Loader",
            {
                "High": extract_filename(high),
                "Low": extract_filename(low),
                "Paired": low != high,
                "TextEncoder": extract_filename(text_encoder),
                "Vae": extract_filename(vae),
            },
        )
        return (model_high, model_low, clip, loaded_vae, info)


def _load_text_encoder(text_encoder: str):
    encoder_path = get_path_by_filename(text_encoder, FOLDER_TEXT_ENCODERS)
    if encoder_path is None:
        raise ValueError(f"Text encoder not found: {text_encoder}. Wan wants umt5, in models/text_encoders.")

    clip, = CLIPLoader().load_clip(encoder_path, DEFAULT_CLIP_TYPE)
    return clip


def _load_vae(vae: str):
    vae_path = get_path_by_filename(vae, FOLDER_VAE)
    if vae_path is None:
        raise ValueError(f"VAE not found: {vae}. Every 14B Wan model reads 16-channel latents, so it wants the 2.1 vae.")

    loaded_vae, = VAELoader().load_vae(vae_path)
    return loaded_vae


def _load_checkpoint(checkpoint: str):
    checkpoint_path = get_path_by_filename(checkpoint, FOLDER_CHECKPOINTS)
    if checkpoint_path is None:
        raise ValueError(f"Checkpoint not found: {checkpoint}")

    return CheckpointLoaderSimple().load_checkpoint(checkpoint_path)
