from __future__ import annotations

from ...registry import NodeRegistration, register_nodes as _register_nodes

from .primitives import Bool, Int, Float, String, Text
from .seed import Seed
from .prompt import ModelConfigNode, PromptConfigNode, VideoModelConfigNode
from .checkpoint import CustomCheckpointLoader, WanCheckpointLoader, WanExpertCheckpointLoader
from .loras import CustomLoraLoader, LorasLoader, WanExpertLoras
from .list import (
    CheckpointList,
    FloatList,
    IntList,
    LorasList,
    SamplerList,
    SchedulerList,
    StringList,
    DimensionsList,
    VideoDimensionsList,
    UpscaleLatentModeList,
    ImageResizeMethodList,
    PromptSeparatorList,
    PositiveStylePositionList,
)
from .math import CompareFloat, CompareInt, MathFloat, MathInt
from .range import Range, RangeList, XYRange, XYZRange
from .switch import AnySwitch, AnyTypeSwitch, FloatSwitch, IntSwitch, StringSwitch
from .convert import FloatToInt, IntToFloat, ToString
from .sizes import GetImageSize, GetLatentSize
from .string import (
    StringConcat,
    StringEquals,
    StringLength,
    StringReplace,
    ToStringConcat,
)
from .display import DisplayAny
from .reroute import Reroute
from .muter import GroupsMuter
from .upscale import UpscaleLatentBy
from .image_save import CustomImageSave
from .image_preview import ImagePreview
from .image import ImageBlank, ImagePowerNoise, ImageToLatent, LatentToImage, LoadImageFromPath
from .image_grid import XYImageGrid, XYZImageGrid
from .image_filters import (
    GaussianBlurMask,
    ImageAdjustment,
    ImageBlendMask,
    ImageBlendMode,
    ImageCannyFilter,
    ImageDragonFilter,
    ImageHighPassFilter,
    ImageLevels,
    ImageLucySharpen,
    ImagePixelate,
    ImageStyleFilter,
)
from .detailer import DetailerStep
from .detect import DetectRegions
from .pipeline import GenerateStep, HiresStep, UpscaleStep, VideoUpscaleStep, VideoLatentUpscaleStep, WanImageToVideoStep, WanVaceToVideoStep
from .rembg import BiRefNetRemBG, BiRefNetRemBGAdvanced
from .video_save import CustomVideoSave
from .video_preview import VideoPreview
from .video import VideoDecode


def register_nodes() -> None:
    _register_nodes([
        NodeRegistration("Bool", Bool, "Bool Value"),
        NodeRegistration("Int", Int, "Int Value"),
        NodeRegistration("Float", Float, "Float Value"),
        NodeRegistration("String", String, "String Value"),
        NodeRegistration("Text", Text, "Text Value"),
        NodeRegistration("Seed", Seed, "Seed Value"),
        NodeRegistration("PromptConfig", PromptConfigNode, "Prompt Config"),
        NodeRegistration("ModelConfig", ModelConfigNode, "Model Config"),
        NodeRegistration("VideoModelConfig", VideoModelConfigNode, "Video Model Config"),
        NodeRegistration("CheckpointLoader", CustomCheckpointLoader, "Load Checkpoint"),
        NodeRegistration("LorasLoader", LorasLoader, "Load Loras"),
        NodeRegistration("LoraLoader", CustomLoraLoader, "Load Lora"),
        NodeRegistration("WanExpertLoras", WanExpertLoras, "Load Wan Expert Loras"),
        NodeRegistration("IntList", IntList, "Int List"),
        NodeRegistration("FloatList", FloatList, "Float List"),
        NodeRegistration("StringList", StringList, "String List"),
        NodeRegistration("CheckpointList", CheckpointList, "Checkpoint List"),
        NodeRegistration("SamplerList", SamplerList, "Sampler List"),
        NodeRegistration("SchedulerList", SchedulerList, "Scheduler List"),
        NodeRegistration("LorasList", LorasList, "Loras List"),
        NodeRegistration("DimensionsList", DimensionsList, "Dimensions List"),
        NodeRegistration("VideoDimensionsList", VideoDimensionsList, "Video Dimensions List"),
        NodeRegistration("UpscaleLatentModeList", UpscaleLatentModeList, "Upscale Latent Mode List"),
        NodeRegistration("ImageResizeMethodList", ImageResizeMethodList, "Image Resize Method List"),
        NodeRegistration("PromptSeparatorList", PromptSeparatorList, "Prompt Separator List"),
        NodeRegistration("PositiveStylePositionList", PositiveStylePositionList, "Positive Style Position List"),
        NodeRegistration("MathInt", MathInt, "Int Math"),
        NodeRegistration("MathFloat", MathFloat, "Float Math"),
        NodeRegistration("CompareInt", CompareInt, "Int Compare"),
        NodeRegistration("CompareFloat", CompareFloat, "Float Compare"),
        NodeRegistration("Range", Range, "Range"),
        NodeRegistration("XYRange", XYRange, "XY Range"),
        NodeRegistration("XYZRange", XYZRange, "XYZ Range"),
        NodeRegistration("RangeList", RangeList, "Range List"),
        NodeRegistration("AnySwitch", AnySwitch, "Any Switch"),
        NodeRegistration("AnyTypeSwitch", AnyTypeSwitch, "AnyType Switch"),
        NodeRegistration("IntSwitch", IntSwitch, "Int Switch"),
        NodeRegistration("FloatSwitch", FloatSwitch, "Float Switch"),
        NodeRegistration("StringSwitch", StringSwitch, "String Switch"),
        NodeRegistration("StringLength", StringLength, "String Length"),
        NodeRegistration("StringConcat", StringConcat, "String Concat"),
        NodeRegistration("StringReplace", StringReplace, "String Replace"),
        NodeRegistration("StringEquals", StringEquals, "String Equals"),
        NodeRegistration("ToStringConcat", ToStringConcat, "To String Concat"),
        NodeRegistration("IntToFloat", IntToFloat, "Int To Float"),
        NodeRegistration("FloatToInt", FloatToInt, "Float To Int"),
        NodeRegistration("ToString", ToString, "To String"),
        NodeRegistration("GetLatentSize", GetLatentSize, "Latent Size"),
        NodeRegistration("GetImageSize", GetImageSize, "Image Size"),
        NodeRegistration("Reroute", Reroute, "Reroute"),
        NodeRegistration("GroupsMuter", GroupsMuter, "Groups Muter"),
        NodeRegistration("DisplayAny", DisplayAny, "Display Any"),
        NodeRegistration("UpscaleLatentBy", UpscaleLatentBy, "Upscale Latent By Model"),
        NodeRegistration("ImageBlank", ImageBlank, "Image Blank"),
        NodeRegistration("ImagePowerNoise", ImagePowerNoise, "Image Power Noise"),
        NodeRegistration("LoadImageFromPath", LoadImageFromPath, "Load Image From Path"),
        NodeRegistration("ImageToLatent", ImageToLatent, "Image To Latent"),
        NodeRegistration("LatentToImage", LatentToImage, "Latent To Image"),
        NodeRegistration("SaveImage", CustomImageSave, "Save Image"),
        NodeRegistration("PreviewImage", ImagePreview, "Preview Image"),
        NodeRegistration("SaveVideo", CustomVideoSave, "Save Video"),
        NodeRegistration("PreviewVideo", VideoPreview, "Preview Video"),
        NodeRegistration("XYImageGrid", XYImageGrid, "XY Image Grid"),
        NodeRegistration("XYZImageGrid", XYZImageGrid, "XYZ Image Grid"),
        NodeRegistration("ImagePixelate", ImagePixelate, "Image Pixelate"),
        NodeRegistration("ImageAdjustment", ImageAdjustment, "Image Adjustment"),
        NodeRegistration("ImageLucySharpen", ImageLucySharpen, "Image Lucy Sharpen"),
        NodeRegistration("ImageStyleFilter", ImageStyleFilter, "Image Style Filter"),
        NodeRegistration("ImageHighPassFilter", ImageHighPassFilter, "Image High Pass Filter"),
        NodeRegistration("ImageLevels", ImageLevels, "Image Levels"),
        NodeRegistration("ImageCannyFilter", ImageCannyFilter, "Image Canny Filter"),
        NodeRegistration("ImageDragonFilter", ImageDragonFilter, "Image Dragon Filter"),
        NodeRegistration("ImageBlendMode", ImageBlendMode, "Image Blend Mode"),
        NodeRegistration("ImageBlendMask", ImageBlendMask, "Image Blend Mask"),
        NodeRegistration("GaussianBlurMask", GaussianBlurMask, "Gaussian Blur Mask"),
        NodeRegistration("GenerateStep", GenerateStep, "Generate Step"),
        NodeRegistration("HiresStep", HiresStep, "Hires Step"),
        NodeRegistration("UpscaleStep", UpscaleStep, "Upscale Step"),
        NodeRegistration("DetailerStep", DetailerStep, "Detailer Step"),
        NodeRegistration("DetectRegions", DetectRegions, "Detect Regions"),
        NodeRegistration("VideoUpscaleStep", VideoUpscaleStep, "Video Upscale Step"),
        NodeRegistration("VideoLatentUpscaleStep", VideoLatentUpscaleStep, "Video Latent Upscale Step"),
        NodeRegistration("WanImageToVideoStep", WanImageToVideoStep, "Wan Image To Video Step"),
        NodeRegistration("WanVaceToVideoStep", WanVaceToVideoStep, "Wan VACE To Video Step"),
        NodeRegistration("WanCheckpointLoader", WanCheckpointLoader, "Wan Checkpoint Loader"),
        NodeRegistration("WanExpertCheckpointLoader", WanExpertCheckpointLoader, "Wan Expert Checkpoint Loader"),
        NodeRegistration("VideoDecode", VideoDecode, "Video Decode"),
        NodeRegistration("BRIARemBG", BiRefNetRemBG, "BRIA Rem BG"),
        NodeRegistration("BRIARemBGAdvanced", BiRefNetRemBGAdvanced, "BRIA Rem BG Advanced"),
    ])
