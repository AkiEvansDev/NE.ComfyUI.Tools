from __future__ import annotations

import os

import torch
from comfy import model_management

from ...constants import *
from ..core.latent_resizer import LatentResizer
from .helpers import Input

# One resizer per (path, dtype) for the whole process. HiresStep builds an UpscaleLatentBy for every pass, so a
# model held by the instance was read from disk again on every custom-mode hires run.
_resizers: dict[tuple[str, torch.dtype], LatentResizer] = {}


class UpscaleLatentBy:
    def __init__(self):
        self.scale_factor = LATENT_RESIZER_SCALE_FACTOR
        self.dtype = torch.float16 if model_management.should_use_fp16() else torch.float32
        self.model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
            *LATENT_RESIZER_MODEL_RELATIVE_PATH,
        )

    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(
            latent=("LATENT",),
            scale_by=Input.float(DEFAULT_LATENT_SCALE_BY, LATENT_SCALE_BY_MIN, LATENT_SCALE_BY_MAX, LATENT_SCALE_BY_STEP),
        )

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "upscale"
    CATEGORY = CATEGORY_IMAGE

    def upscale(self, latent, scale_by: float):
        scale_by = round(scale_by, 2)

        device = model_management.get_torch_device()
        samples = latent["samples"].to(device=device, dtype=self.dtype)

        key = (self.model_path, self.dtype)
        if key not in _resizers:
            _resizers[key] = LatentResizer.load_model(self.model_path, device, self.dtype)

        model = _resizers[key].to(device=device)
        latent_out = model(self.scale_factor * samples, scale=scale_by) / self.scale_factor

        if self.dtype != torch.float32:
            latent_out = latent_out.to(dtype=torch.float32)

        latent_out = latent_out.to(device="cpu")
        model.to(device=model_management.vae_offload_device())

        result = latent.copy()
        result["samples"] = latent_out

        return (result,)
