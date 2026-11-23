# Changelog

This file holds only what is not released yet, under `## X.Y.Z`: the release workflow cuts that section out as
the body of the GitHub release (a tag with no section fails the release), and the notes of every released
version live there — https://github.com/AkiEvansDev/NE.ComfyUI.Tools/releases.

## 1.0.0

- Initial public release: 89 `NE.Tools` nodes and 18 frontend extensions for ComfyUI.
- Typed step pipelines (`Generate Step`, `Hires Step`, `Detailer Step`, `Upscale Step`) driven by `PROMPT_CONFIG`, `MODEL_CONFIG`, `SAMPLER_CONFIG` and `PIPELINE_INFO` bundles.
- GPU YOLOv8 + MobileSAM region detection (`Detect Regions`) and region-isolated detailing with `DifferentialDiffusion` and numbered per-region prompts.
- Wan 2.1 I2V and Wan 2.2 14B high/low dual-expert video pipelines (`Wan Image To Video Step`, `Wan VACE To Video Step`, `Video Latent Upscale Step`).
- Stateful `Range` / `XY Range` / `XYZ Range` counters and `XY Image Grid` / `XYZ Image Grid` composers.
- Image filters, BiRefNet background removal, dynamic LoRA stacks, booru tag autocomplete, `Groups Muter`, typed `Reroute`, `Display Any`, and execution progress bar.
