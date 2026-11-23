from __future__ import annotations

import math
import random
from collections.abc import Callable

import comfy.model_management
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from ...constants import *
from ..core.deps import require_dependency
from ..core.image import ensure_mask_batch, frame_at, pil2tensor, process_batch, tensor2pil
from .helpers import Input

ImageProcessor = Callable[[torch.Tensor], torch.Tensor]

def image_input():
    return Input.force("IMAGE")


def mask_input():
    return Input.force("MASK")


def batch(images: torch.Tensor, processor: ImageProcessor) -> torch.Tensor:
    return process_batch(images, processor)


def round2(value: float) -> float:
    return round(value, 2)


def solid_mask(size: tuple[int, int], strength: float) -> Image.Image:
    return Image.new("L", size, round(strength * 255))


def resize_like(image: Image.Image, reference: Image.Image) -> Image.Image:
    return image.resize(reference.size)


def to_rgb(image: torch.Tensor) -> Image.Image:
    return tensor2pil(image).convert("RGB")


def tensor_from_rgb(image: Image.Image) -> torch.Tensor:
    return pil2tensor(image.convert("RGB"))


def palette_colors(image: Image.Image, num_colors: int, init_mode: str) -> np.ndarray:
    """The (N, 3) uint8 palette a pixelated image is reduced to: k-means centres, or a median cut without them."""
    pixels = np.array(image).reshape(-1, 3)
    # Never more clusters than distinct colours: k-means refuses outright, and a small or flat image has few.
    count = min(num_colors, len(np.unique(pixels, axis=0)))

    if init_mode == PIXEL_INIT_NONE:
        quantized = image.quantize(colors=count, method=Image.Quantize.MEDIANCUT)
        return np.array(quantized.getpalette()[:count * 3], dtype=np.uint8).reshape(-1, 3)

    sklearn_cluster = require_dependency("sklearn.cluster", "scikit-learn")
    kmeans = sklearn_cluster.KMeans(
        n_clusters=count,
        init=init_mode,
        max_iter=100,
        tol=1e-3,
        random_state=42,
        n_init="auto",
    )
    kmeans.fit(pixels)
    return np.clip(np.round(kmeans.cluster_centers_), 0, 255).astype(np.uint8)


def reduce_colors(image: Image.Image, colors: np.ndarray, dither_mode: str | None) -> Image.Image:
    """Every pixel mapped to its nearest palette colour, dithered or not. The mapping runs inside PIL, in C."""
    palette = _palette_image(colors)

    if dither_mode == DITHER_MODE_FLOYD:
        return image.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG).convert("RGB")

    if dither_mode == DITHER_MODE_ORDERED:
        image = _bayer_offset(image, colors)

    return image.quantize(palette=palette, dither=Image.Dither.NONE).convert("RGB")


def _palette_image(colors: np.ndarray) -> Image.Image:
    # PIL matches against all 256 entries, so the unused ones repeat the first colour rather than stay black.
    entries = np.concatenate([colors, np.repeat(colors[:1], 256 - len(colors), axis=0)])
    palette = Image.new("P", (1, 1))
    palette.putpalette(entries.reshape(-1).tolist())
    return palette


def _bayer_offset(image: Image.Image, colors: np.ndarray) -> Image.Image:
    """Ordered dithering: a tiled Bayer threshold nudges each pixel before the nearest-colour pick."""
    bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], dtype=np.float32) / 16.0 - 0.5
    array = np.array(image, dtype=np.float32)
    height, width = array.shape[:2]
    threshold = np.tile(bayer, (height // 4 + 1, width // 4 + 1))[:height, :width, None]
    # One palette step's worth of spread, measured on the palette itself: the typical distance from a colour to
    # its nearest neighbour. Assuming the colours sat evenly across the whole cube threw a light image, whose
    # palette is packed into the top of it, a nudge several steps wide and turned it into a coarse grid.
    points = colors.astype(np.float32)
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    spread = float(np.median(distances.min(axis=1))) if len(colors) > 1 else 0.0
    return Image.fromarray(np.clip(array + threshold * spread, 0, 255).astype(np.uint8))


def apply_instagram_style(image: Image.Image, style: str) -> Image.Image:
    pilgram = require_dependency("pilgram")

    if style == "fairy tale":
        return apply_fairy_tale_style(image)
    if style == "aden":
        return apply_aden_style(image)
    if style == "1977":
        return pilgram._1977(image)

    return getattr(pilgram, style.replace(" ", "_"))(image)


def apply_aden_style(image: Image.Image) -> Image.Image:
    """pilgram's aden with its gradient built for the image's own shape.

    pilgram makes the horizontal gradient by rotating a mask of the image's size by 90 degrees without
    expanding it, so on any frame that is not square the mask is turned inside its own rectangle and a band
    of the picture came out as a hard-edged box. The steps and numbers are pilgram's; only the mask differs.
    """
    pilgram = require_dependency("pilgram")
    base = image.convert("RGB")
    width, height = base.size

    tinted = pilgram.css.blending.darken(base, Image.new("RGB", base.size, (66, 10, 14)))
    # Built at the transposed size and rotated with expand, so it lands on the image's size, left to right.
    gradient = ImageOps.invert(Image.linear_gradient("L"))
    box = (0, round(gradient.height * 0.8), gradient.width, gradient.height)
    mask = gradient.resize((height, width), box=box).rotate(90, expand=True)

    result = Image.composite(tinted, base, mask)
    result = pilgram.css.hue_rotate(result, -20)
    result = pilgram.css.contrast(result, 0.9)
    result = pilgram.css.saturate(result, 0.85)
    return pilgram.css.brightness(result, 1.2)


def apply_fairy_tale_style(image: Image.Image) -> Image.Image:
    pilgram = require_dependency("pilgram")
    image = image.convert("RGBA")
    image = ImageEnhance.Contrast(image).enhance(1.25)
    image = ImageEnhance.Color(image).enhance(1.5)

    bloom = image.filter(ImageFilter.GaussianBlur(radius=20))
    bloom = ImageEnhance.Brightness(bloom).enhance(1.2)
    bloom.putalpha(128)
    image = Image.alpha_composite(image, bloom.convert(image.mode))

    particles_a = create_particles(*image.size)
    particles_b = create_particles(*image.size)

    image = pilgram.css.blending.color_dodge(image, particles_a)
    return pilgram.css.blending.lighten(image, particles_b)


def create_particles(width: int, height: int) -> Image.Image:
    particles = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(particles)

    for _ in range(5000):
        draw.point(
            (random.randint(0, width - 1), random.randint(0, height - 1)),
            fill=(
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
                255,
            ),
        )

    particles = particles.filter(ImageFilter.GaussianBlur(radius=1))
    particles.putalpha(128)

    return particles


def high_pass_image(
    image: Image.Image,
    radius: float,
    strength: float,
    color_output: bool,
    neutral_background: bool,
) -> Image.Image:
    source = image.convert("RGB")
    source_arr = np.array(source).astype("float")
    blurred_arr = np.array(source.filter(ImageFilter.GaussianBlur(radius=radius))).astype("float")
    # On a neutral background the detail is centred on grey. Clipped at zero first and screened over grey after,
    # every darker-than-surroundings detail vanished and a dark dot came out as flat grey.
    offset = HIGH_PASS_NEUTRAL_LEVEL if neutral_background else 0.0
    high_pass_arr = np.clip((source_arr - blurred_arr) * strength + offset, 0, 255).astype("uint8")

    high_pass = (
        Image.fromarray(high_pass_arr, mode="RGB")
        if color_output
        else Image.fromarray(np.mean(high_pass_arr, axis=2).astype("uint8"), mode="L")
    )

    return high_pass.convert("RGB")


def apply_levels(image: Image.Image, black_level: float, mid_level: float, white_level: float) -> Image.Image:
    if white_level <= black_level:
        white_level = black_level + 1

    image_arr = np.array(image).astype(np.float32)
    image_arr[image_arr < black_level] = black_level
    image_arr = np.clip((image_arr - black_level) * (255 / (white_level - black_level)), 0, 255)

    # Outside (black, white) the midpoint has no gamma: at white it is log(1) = 0 in the denominator, and past
    # it the gamma turns negative and the picture inverts into infinities.
    if not black_level < mid_level < white_level:
        gamma = 1.0
    else:
        gamma = math.log(0.5) / math.log((mid_level - black_level) / (white_level - black_level))

    image_arr = np.power(image_arr / 255, gamma) * 255
    return Image.fromarray(image_arr.astype(np.uint8))


def laplacian_edges(image: Image.Image) -> Image.Image:
    return image.filter(
        ImageFilter.Kernel(
            (3, 3),
            (-1, -1, -1, -1, 8, -1, -1, -1, -1),
            1,
            0,
        ),
    )


def canny_edges(image: torch.Tensor) -> torch.Tensor:
    cv2 = require_dependency("cv2", "opencv-python")
    array = (255.0 * image.detach().cpu().numpy()).astype(np.uint8)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    return tensor_from_rgb(Image.fromarray(edges))


def dragan_image(
    image: Image.Image,
    saturation: float,
    contrast: float,
    sharpness: float,
    brightness: float,
    highpass_samples: int,
    highpass_strength: float,
    highpass_radius: float,
    colorize: bool,
) -> Image.Image:
    pilgram = require_dependency("pilgram")
    source = image.convert("RGB")

    grayscale = source.convert("L")
    filtered = ImageEnhance.Contrast(grayscale).enhance(contrast)
    filtered = ImageEnhance.Color(filtered).enhance(saturation)
    filtered = ImageEnhance.Sharpness(filtered).enhance(sharpness)
    filtered = ImageEnhance.Brightness(filtered).enhance(brightness)

    blurred = filtered.filter(ImageFilter.GaussianBlur(radius=highpass_radius))
    highpass = ImageChops.subtract(source, blurred.convert("RGB"))
    neutral = Image.new("RGB", source.size, (127, 127, 127))
    highpass = ImageChops.screen(neutral, highpass.resize(source.size))

    if not colorize:
        highpass = highpass.convert("L").convert("RGB")

    filtered_rgb = filtered.convert("RGB")
    highpassed = pilgram.css.blending.overlay(filtered_rgb, highpass)

    for _ in range(max(highpass_samples, 1)):
        highpassed = pilgram.css.blending.overlay(highpassed, highpass)

    final = ImageChops.blend(filtered_rgb, highpassed, highpass_strength)

    if colorize:
        final = pilgram.css.blending.color(final, source)

    return final.convert("RGB")


def blend_images(image_a: Image.Image, image_b: Image.Image, mode: str, opacity: float) -> Image.Image:
    pilgram = require_dependency("pilgram")
    image_a = image_a.convert("RGB")
    image_b = resize_like(image_b.convert("RGB"), image_a)

    # pilgram has no add mode; standing in `normal` for it returned image_b unchanged.
    blended = (
        ImageChops.add(image_a, image_b)
        if mode == BLEND_MODE_ADD
        else getattr(pilgram.css.blending, mode)(image_a, image_b)
    ).convert("RGB")

    return Image.composite(image_a, blended, ImageOps.invert(solid_mask(image_a.size, opacity)))


def mask_to_pil(mask: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    # One frame only: tensor2pil goes through Image.fromarray, which cannot take a batch dimension.
    frame = mask.reshape(mask.shape[-2], mask.shape[-1])
    return ImageOps.invert(tensor2pil(frame).convert("L")).resize(size)


def normalize_mask(mask: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)

    if mask.ndim == 2:
        return mask[None, ..., None]
    if mask.ndim == 3:
        return mask[..., None]
    if mask.ndim == 4:
        return mask

    raise ValueError(f"Expected 2D, 3D, or 4D mask tensor, but found {mask.ndim} dimensions.")


def blur_mask_tensor(mask: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    mask = normalize_mask(mask)

    if mask.shape[-1] != 1:
        raise ValueError(f"Expected 1 channel for mask, but found {mask.shape[-1]} channels.")

    if kernel_size <= 0:
        return mask

    kernel_size = kernel_size * 2 + 1
    shortest = min(mask.shape[1], mask.shape[2])

    if shortest <= kernel_size:
        kernel_size = int(shortest / 2)
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 3:
            return mask

    # Separable: two 1D passes instead of the full 2D kernel torchvision builds, which at the radii the detailer
    # reaches (a noise feather scaled up to working size, ~170px) took 170ms a call against 3ms. The trip to
    # the GPU and back is not the cost - measured at a few milliseconds - and on the CPU the same convolution
    # holds every core, so it still runs on the card.
    previous_device = mask.device
    device = comfy.model_management.get_torch_device()
    planes = mask.to(device)[:, None, ..., 0]

    offsets = torch.arange(kernel_size, dtype=planes.dtype, device=device) - kernel_size // 2
    weights = torch.exp(-(offsets ** 2) / (2.0 * sigma ** 2))
    weights = weights / weights.sum()
    pad = kernel_size // 2

    # Reflect at the border, as torchvision does: a mask touching the frame keeps its weight there.
    planes = F.conv2d(F.pad(planes, (pad, pad, 0, 0), mode="reflect"), weights.view(1, 1, 1, -1))
    planes = F.conv2d(F.pad(planes, (0, 0, pad, pad), mode="reflect"), weights.view(1, 1, -1, 1))
    return planes[:, 0, ..., None].to(previous_device)


class ImagePixelate:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(
            images=image_input(),
            pixelation_size=Input.int(DEFAULT_PIXELATE_SIZE, PIXELATE_SIZE_MIN, PIXELATE_SIZE_MAX),
            num_colors=Input.int(DEFAULT_PIXELATE_COLORS, PIXELATE_COLORS_MIN, PIXELATE_COLORS_MAX),
            init_mode=Input.choice(PIXEL_INIT_MODES, DEFAULT_PIXEL_INIT_MODE),
            dither=Input.bool(DEFAULT_PIXELATE_DITHER),
            dither_mode=Input.choice(DITHER_MODES, DEFAULT_DITHER_MODE),
        )

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY_IMAGE_FILTER

    def execute(self, images, pixelation_size, num_colors, init_mode, dither, dither_mode):
        def process(image: torch.Tensor) -> torch.Tensor:
            return self.pixelate(image, pixelation_size, num_colors, init_mode, dither, dither_mode)

        return (batch(images, process),)

    def pixelate(self, image: torch.Tensor, target_size: int, num_colors: int, init_mode: str, dither: bool, dither_mode: str) -> torch.Tensor:
        pil_image = to_rgb(image)
        original_width, original_height = pil_image.size

        if max(original_width, original_height) > target_size:
            scale = target_size / max(original_width, original_height)
            pil_image = pil_image.resize(
                (int(original_width * scale), int(original_height * scale)),
                Image.Resampling.NEAREST,
            )

        # The palette is found first and the image mapped onto it once: dithering an image already flattened
        # to its palette has no error left to spread, and used to quantise to levels of its own instead.
        if init_mode != PIXEL_INIT_NONE or dither:
            colors = palette_colors(pil_image, num_colors, init_mode)
            pil_image = reduce_colors(pil_image, colors, dither_mode if dither else None)

        pil_image = pil_image.resize((original_width, original_height), Image.Resampling.NEAREST)

        return tensor_from_rgb(pil_image)


class ImageAdjustment:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(
            images=image_input(),
            brightness=Input.float(DEFAULT_ADJUST_BRIGHTNESS, ADJUST_BRIGHTNESS_MIN, ADJUST_BRIGHTNESS_MAX, ADJUST_STEP),
            contrast=Input.float(DEFAULT_ADJUST_CONTRAST, ADJUST_CONTRAST_MIN, ADJUST_CONTRAST_MAX, ADJUST_STEP),
            saturation=Input.float(DEFAULT_ADJUST_SATURATION, ADJUST_SATURATION_MIN, ADJUST_SATURATION_MAX, ADJUST_STEP),
            sharpness=Input.float(DEFAULT_ADJUST_SHARPNESS, ADJUST_SHARPNESS_MIN, ADJUST_SHARPNESS_MAX, ADJUST_STEP),
            blur=Input.int(DEFAULT_ADJUST_BLUR, ADJUST_BLUR_MIN, ADJUST_BLUR_MAX),
            gaussian_blur=Input.float(
                DEFAULT_ADJUST_GAUSSIAN_BLUR,
                ADJUST_GAUSSIAN_BLUR_MIN,
                ADJUST_GAUSSIAN_BLUR_MAX,
                ADJUST_GAUSSIAN_BLUR_STEP,
            ),
            edge_enhance=Input.float(DEFAULT_ADJUST_EDGE_ENHANCE, ADJUST_EDGE_ENHANCE_MIN, ADJUST_EDGE_ENHANCE_MAX, ADJUST_STEP),
            detail_enhance=Input.bool(DEFAULT_ADJUST_DETAIL_ENHANCE),
        )

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY_IMAGE_FILTER

    def execute(self, images, brightness, contrast, saturation, sharpness, blur, gaussian_blur, edge_enhance, detail_enhance):
        brightness = round2(brightness)
        contrast = round2(contrast)
        saturation = round2(saturation)
        sharpness = round2(sharpness)
        gaussian_blur = round2(gaussian_blur)
        edge_enhance = round2(edge_enhance)

        def process(image: torch.Tensor) -> torch.Tensor:
            pil_image: Image.Image | None = None

            if brightness != 0.0:
                image = torch.clamp(image + brightness, 0.0, 1.0)

            # About mid-grey: scaling the value itself is a gain, which brightened at 2.0 and blacked out at 0.
            if contrast != 1.0:
                image = torch.clamp((image - 0.5) * contrast + 0.5, 0.0, 1.0)

            if saturation != 1.0:
                pil_image = ImageEnhance.Color(to_rgb(image)).enhance(saturation)

            if sharpness != 1.0:
                pil_image = ImageEnhance.Sharpness(pil_image or to_rgb(image)).enhance(sharpness)

            if blur > 0:
                pil_image = pil_image or to_rgb(image)
                for _ in range(blur):
                    pil_image = pil_image.filter(ImageFilter.BLUR)

            if gaussian_blur > 0.0:
                pil_image = (pil_image or to_rgb(image)).filter(ImageFilter.GaussianBlur(radius=gaussian_blur))

            if edge_enhance > 0.0:
                pil_image = pil_image or to_rgb(image)
                edge_image = pil_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
                pil_image = Image.composite(edge_image, pil_image, solid_mask(pil_image.size, edge_enhance))

            if detail_enhance:
                pil_image = (pil_image or to_rgb(image)).filter(ImageFilter.DETAIL)

            return tensor_from_rgb(pil_image) if pil_image else image.unsqueeze(0)

        return (batch(images, process),)


class ImageLucySharpen:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(
            images=image_input(),
            iterations=Input.int(DEFAULT_LUCY_ITERATIONS, LUCY_ITERATIONS_MIN, LUCY_ITERATIONS_MAX),
            kernel_size=Input.int(DEFAULT_LUCY_KERNEL_SIZE, LUCY_KERNEL_SIZE_MIN, LUCY_KERNEL_SIZE_MAX),
        )

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY_IMAGE_FILTER

    def execute(self, images, iterations, kernel_size):
        return (batch(images, lambda image: self.sharpen(image, iterations, kernel_size)),)

    def sharpen(self, image: torch.Tensor, iterations: int, kernel_size: int) -> torch.Tensor:
        """Richardson-Lucy deconvolution against a box blur of `kernel_size`.

        The ratio is the *observed* image over the blurred estimate. Dividing the estimate by its own blur, as
        this used to, is not deconvolution and does not converge; with zero padding at the border on top of it,
        a flat grey came back dark in the corners and ringing a few rows in, and collapsed to black at k=16.
        """
        scipy_ndimage = require_dependency("scipy.ndimage", "scipy")

        observed = np.array(to_rgb(image), dtype=np.float32) / 255.0
        size = (kernel_size, kernel_size, 1)
        # A box is its own mirror, but an even one is not centred: the adjoint pass leans the other way, or every
        # iteration drifts the picture by a pixel.
        back = -1 if kernel_size % 2 == 0 else 0

        estimate = observed.copy()
        for _ in range(iterations):
            blurred = scipy_ndimage.uniform_filter(estimate, size=size, mode="reflect")
            ratio = observed / (blurred + 1e-6)
            estimate *= scipy_ndimage.uniform_filter(ratio, size=size, mode="reflect", origin=(back, back, 0))

        sharpened = np.clip(estimate * 255.0, 0, 255).astype(np.uint8)
        return tensor_from_rgb(Image.fromarray(sharpened))


class ImageStyleFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(
            images=image_input(),
            style=Input.choice(STYLE_FILTERS, DEFAULT_STYLE_FILTER),
        )

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY_IMAGE_FILTER

    def execute(self, images, style):
        return (batch(images, lambda image: tensor_from_rgb(apply_instagram_style(to_rgb(image), style))),)


class ImageHighPassFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(
            images=image_input(),
            radius=Input.int(DEFAULT_HIGH_PASS_RADIUS, HIGH_PASS_RADIUS_MIN, HIGH_PASS_RADIUS_MAX),
            strength=Input.float(DEFAULT_HIGH_PASS_STRENGTH, HIGH_PASS_STRENGTH_MIN, HIGH_PASS_STRENGTH_MAX, HIGH_PASS_STRENGTH_STEP),
            color_output=Input.bool(DEFAULT_HIGH_PASS_COLOR_OUTPUT),
            neutral_background=Input.bool(DEFAULT_HIGH_PASS_NEUTRAL_BACKGROUND),
        )

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY_IMAGE_FILTER

    def execute(self, images, radius, strength, color_output, neutral_background):
        strength = round2(strength)
        return (batch(images, lambda image: tensor_from_rgb(high_pass_image(to_rgb(image), radius, strength, color_output, neutral_background))),)


class ImageLevels:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(
            images=image_input(),
            black_level=Input.float(DEFAULT_LEVELS_BLACK, LEVELS_MIN, LEVELS_MAX, LEVELS_STEP),
            mid_level=Input.float(DEFAULT_LEVELS_MID, LEVELS_MIN, LEVELS_MAX, LEVELS_STEP),
            white_level=Input.float(DEFAULT_LEVELS_WHITE, LEVELS_MIN, LEVELS_MAX, LEVELS_STEP),
        )

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY_IMAGE_FILTER

    def execute(self, images, black_level, mid_level, white_level):
        black_level = round2(black_level)
        mid_level = round2(mid_level)
        white_level = round2(white_level)

        return (batch(images, lambda image: tensor_from_rgb(apply_levels(to_rgb(image), black_level, mid_level, white_level))),)


class ImageCannyFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(
            images=image_input(),
            mode=Input.choice(CANNY_MODES, DEFAULT_CANNY_MODE),
        )

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY_IMAGE_FILTER

    def execute(self, images, mode):
        def process(image: torch.Tensor) -> torch.Tensor:
            if mode == "laplacian":
                return tensor_from_rgb(laplacian_edges(to_rgb(image)))
            return canny_edges(image)

        return (batch(images, process),)


class ImageDragonFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(
            images=image_input(),
            saturation=Input.float(DEFAULT_DRAGON_SATURATION, DRAGON_MIN, DRAGON_MAX, DRAGON_STEP),
            contrast=Input.float(DEFAULT_DRAGON_CONTRAST, DRAGON_MIN, DRAGON_MAX, DRAGON_STEP),
            brightness=Input.float(DEFAULT_DRAGON_BRIGHTNESS, DRAGON_MIN, DRAGON_MAX, DRAGON_STEP),
            sharpness=Input.float(DEFAULT_DRAGON_SHARPNESS, DRAGON_MIN, DRAGON_SHARPNESS_MAX, DRAGON_STEP),
            highpass_samples=Input.int(DEFAULT_DRAGON_HIGHPASS_SAMPLES, DRAGON_HIGHPASS_SAMPLES_MIN, DRAGON_HIGHPASS_SAMPLES_MAX),
            highpass_strength=Input.float(DEFAULT_DRAGON_HIGHPASS_STRENGTH, DRAGON_MIN, DRAGON_HIGHPASS_STRENGTH_MAX, DRAGON_STEP),
            highpass_radius=Input.float(DEFAULT_DRAGON_HIGHPASS_RADIUS, DRAGON_MIN, DRAGON_HIGHPASS_RADIUS_MAX, DRAGON_STEP),
            colorize=Input.bool(DEFAULT_DRAGON_COLORIZE),
        )

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY_IMAGE_FILTER

    def execute(self, images, saturation, contrast, sharpness, brightness, highpass_samples, highpass_strength, highpass_radius, colorize):
        saturation = round2(saturation)
        contrast = round2(contrast)
        sharpness = round2(sharpness)
        brightness = round2(brightness)
        highpass_strength = round2(highpass_strength)
        highpass_radius = round2(highpass_radius)

        def process(image: torch.Tensor) -> torch.Tensor:
            return tensor_from_rgb(
                dragan_image(
                    to_rgb(image),
                    saturation,
                    contrast,
                    sharpness,
                    brightness,
                    highpass_samples,
                    highpass_strength,
                    highpass_radius,
                    colorize,
                ),
            )

        return (batch(images, process),)


class ImageBlendMode:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(
            image_a=image_input(),
            image_b=image_input(),
            mode=Input.choice(BLEND_MODES, DEFAULT_BLEND_MODE),
            opacity=Input.float(DEFAULT_BLEND_MODE_OPACITY, OPACITY_MIN, OPACITY_MAX, OPACITY_STEP),
        )

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY_IMAGE

    def execute(self, image_a, image_b, mode, opacity):
        opacity = round2(opacity)

        def process(index: int) -> torch.Tensor:
            first = to_rgb(image_a[index])
            second = to_rgb(frame_at(image_b, index))
            return tensor_from_rgb(blend_images(first, second, mode, opacity))

        return (torch.cat([process(index) for index in range(image_a.shape[0])], dim=0),)


class ImageBlendMask:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(
            image_a=image_input(),
            image_b=image_input(),
            mask=mask_input(),
            opacity=Input.float(DEFAULT_BLEND_MASK_OPACITY, OPACITY_MIN, OPACITY_MAX, OPACITY_STEP),
        )

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY_IMAGE

    def execute(self, image_a, image_b, mask, opacity):
        opacity = round2(opacity)
        # An unbatched (H, W) mask would otherwise be indexed by row.
        mask = ensure_mask_batch(mask)

        def process(index: int) -> torch.Tensor:
            first = to_rgb(image_a[index])
            second = resize_like(to_rgb(frame_at(image_b, index)), first)
            image_mask = mask_to_pil(frame_at(mask, index), first.size)

            masked = Image.composite(first, second, image_mask)
            result = Image.composite(first, masked, ImageOps.invert(solid_mask(first.size, opacity)))
            return tensor_from_rgb(result)

        return (torch.cat([process(index) for index in range(image_a.shape[0])], dim=0),)


class GaussianBlurMask:
    @classmethod
    def INPUT_TYPES(cls):
        return Input.required(
            mask=mask_input(),
            kernel_size=Input.int(DEFAULT_MASK_BLUR_KERNEL_SIZE, MASK_BLUR_KERNEL_SIZE_MIN, MASK_BLUR_KERNEL_SIZE_MAX),
            sigma=Input.float(DEFAULT_MASK_BLUR_SIGMA, MASK_BLUR_SIGMA_MIN, MASK_BLUR_SIGMA_MAX, MASK_BLUR_SIGMA_STEP),
        )

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY_IMAGE

    def execute(self, mask, kernel_size, sigma):
        mask = normalize_mask(mask)
        mask = blur_mask_tensor(mask, kernel_size, round2(sigma))
        return (mask.squeeze(-1),)
