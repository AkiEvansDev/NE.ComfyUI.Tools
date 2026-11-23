# Node identity
NODE_PREFIX = "NE."
NODE_CATEGORY_ROOT = "NE.Tools"

# Categories
CATEGORY_ROOT = NODE_CATEGORY_ROOT
CATEGORY_PRIMITIVES = f"{NODE_CATEGORY_ROOT}/Primitive"
CATEGORY_PROMPT_CONFIG = f"{NODE_CATEGORY_ROOT}/Prompt"
CATEGORY_LORA = f"{NODE_CATEGORY_ROOT}/Lora"
CATEGORY_LIST = f"{NODE_CATEGORY_ROOT}/List"
CATEGORY_RANGE = f"{NODE_CATEGORY_ROOT}/Range"
CATEGORY_MATH = f"{NODE_CATEGORY_ROOT}/Math"
CATEGORY_SWITCH = f"{NODE_CATEGORY_ROOT}/Switch"
CATEGORY_CONVERT = f"{NODE_CATEGORY_ROOT}/Convert"
CATEGORY_SIZE = f"{NODE_CATEGORY_ROOT}/Size"
CATEGORY_STRING = f"{NODE_CATEGORY_ROOT}/String"
CATEGORY_CHECKPOINT = f"{NODE_CATEGORY_ROOT}/Checkpoint"
CATEGORY_IMAGE = f"{NODE_CATEGORY_ROOT}/Image"
CATEGORY_IMAGE_FILTER = f"{CATEGORY_IMAGE}/Filter"
CATEGORY_PIPELINE = f"{NODE_CATEGORY_ROOT}/Pipeline"
CATEGORY_REMBG = f"{CATEGORY_IMAGE}/Remove Background"
CATEGORY_DETECT = f"{CATEGORY_IMAGE}/Detect"
CATEGORY_VIDEO = f"{NODE_CATEGORY_ROOT}/Video"

# Server events
SEED_FEEDBACK_EVENT = "ne-seed-node-feedback"
DISPLAY_FEEDBACK_EVENT = "ne-display-node-feedback"
RANGE_FEEDBACK_EVENT = "ne-range-node-feedback"
XY_RANGE_FEEDBACK_EVENT = "ne-xy-range-node-feedback"
XYZ_RANGE_FEEDBACK_EVENT = "ne-xyz-range-node-feedback"

# API routes
API_PREFIX = "/ne"
ROUTE_AUTOCOMPLETE = f"{API_PREFIX}/autocomplete"
ROUTE_LORAS = f"{API_PREFIX}/loras"
# ?experts=1 folds the low-noise half of every Wan 2.2 pair away, so one entry stands for the pair.
QUERY_EXPERTS = "experts"
# What a folded pair is called: the name up to its high/low word, marked as standing for two files.
EXPERT_PAIR_SUFFIX = " (2)"
ROUTE_LORA_INFO = f"{API_PREFIX}/loras/info"
ROUTE_LORA_PREVIEW = f"{API_PREFIX}/loras/preview"
ROUTE_REBOOT = f"{API_PREFIX}/reboot"
REBOOT_HEADER = "X-NE-Reboot"
REBOOT_HEADER_VALUE = "1"
ROUTE_RESET = f"{API_PREFIX}/reset/{{node_id}}"

# Folders
FOLDER_CHECKPOINTS = "checkpoints"
FOLDER_LORAS = "loras"
FOLDER_UPSCALE_MODELS = "upscale_models"
# Preselected when installed; otherwise the first model in the folder is, as before.
DEFAULT_UPSCALE_MODEL = "4x-AnimeSharp"
FOLDER_CLIP_VISION = "clip_vision"
FOLDER_TEXT_ENCODERS = "text_encoders"
FOLDER_VAE = "vae"

# Which text encoder family CLIPLoader is told to build. The list of families is read off the stock node
# rather than copied here: it grows with every model release.
DEFAULT_CLIP_TYPE = "wan"

DEFAULT_VIDEO_POSITIVE_PROMPT = (
    "subtle breathing, gentle blinking, slight hair sway, soft cinematic motion, "
    "minimal movement, stable composition"
)

# File extensions
ALLOWED_IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png", ".tiff", ".gif", ".bmp", ".webp")
SAVE_IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "gif", "tiff", "webp", "bmp"]
SAVE_VIDEO_EXTENSIONS = ["mp4", "webm", "gif"]
DEFAULT_SAVE_VIDEO_EXTENSION = "mp4"
DEFAULT_SAVE_VIDEO_FOLDER = "video/[date]"
DEFAULT_SAVE_VIDEO_PREFIX = "video"
DEFAULT_SAVE_VIDEO_QUALITY = 8
SAVE_VIDEO_QUALITY_MIN = 1
SAVE_VIDEO_QUALITY_MAX = 10
# `quality` 10 and 1 as the encoder's crf, per container: x264 counts 0-51, VP9 0-63. GIF has no crf.
SAVE_VIDEO_CRF_RANGES = {"mp4": (16, 40), "webm": (20, 50)}

# Generic numeric limits
INT_MIN = -0xFFFFFFFFFFFFFFFF
INT_MAX = 0xFFFFFFFFFFFFFFFF
FLOAT_MIN = -0xFFFFFFFFFFFFFFFF
FLOAT_MAX = 0xFFFFFFFFFFFFFFFF
FLOAT_STEP = 0.05
ROUND_DIGITS = 2

# Shared widget limits
MAX_SEED = 9999999999999999
STEPS_MIN = 1
STEPS_MAX = 100
STEPS_STEP = 1
DENOISE_MIN = 0.0
DENOISE_MAX = 1.0
DENOISE_STEP = 0.01
CFG_MIN = 1.0
CFG_MAX = 20.0
CFG_STEP = 0.1
SEED_MIN = 0

# Dimensions
DIMENSIONS = [
    " 768 x 1024 (classic portrait)",
    "1024 x 768  (classic landscape)",
    " 896 x 1152 (balanced portrait)",
    " 832 x 1216 (dramatic portrait)",
    " 768 x 1344 (fashion portrait)",
    " 640 x 1536 (ultra-tall portrait)",
    "1216 x 832  (balanced landscape)",
    "1344 x 768  (cinematic landscape)",
    "1536 x 640  (ultra-wide landscape)",
    "1024 x 1024 (square)",
    "1536 x 1536 (large square)",
    "1024 x 1536 (poster portrait)",
    "1536 x 1024 (poster landscape)",
    "1600 x 900  (cinematic 16:9)",
]
DEFAULT_DIMENSIONS = "1024 x 1536 (poster portrait)"

VIDEO_DIMENSIONS = [
    " 512 x 512  (square)",
    " 640 x 640  (square)",
    " 768 x 768  (square)",
    "1024 x 1024 (square)",
    " 832 x 480  (landscape)",
    " 832 x 640  (landscape)",
    "1024 x 576  (landscape)",
    "1280 x 704  (landscape)",
    "1280 x 720  (landscape)",
    "1920 x 1080 (landscape)",
    " 480 x 832  (portrait)",
    " 576 x 1024 (portrait)",
    " 640 x 832  (portrait)",
    " 704 x 1280 (portrait)",
    " 720 x 1280 (portrait)",
    "1080 x 1920 (portrait)",
]
DEFAULT_VIDEO_DIMENSIONS = " 640 x 640  (square)"

# Sampler defaults
DEFAULT_SAMPLER = "deis"
DEFAULT_SCHEDULER = "beta"
DEFAULT_STEPS = 40
DEFAULT_DENOISE = 1.0
DEFAULT_CFG = 4.0

# Seed defaults
DEFAULT_SEED = 0
SEED_MODE_FIXED = "fixed"
SEED_MODE_RANDOMIZE = "randomize"
SEED_MODE_INCREMENT = "increment"
SEED_MODE_DECREMENT = "decrement"
SEED_MODES = [SEED_MODE_FIXED, SEED_MODE_RANDOMIZE, SEED_MODE_INCREMENT, SEED_MODE_DECREMENT]
DEFAULT_SEED_MODE = SEED_MODE_FIXED

# Hires fix defaults and limits
# The _gpu variant builds its brownian tree on the card; the plain one held all eight cores doing it on the CPU.
DEFAULT_HIRES_SAMPLER = "dpmpp_2m_sde_gpu"
DEFAULT_HIRES_SCHEDULER = "simple"
DEFAULT_HIRES_STEPS = 40
DEFAULT_HIRES_DENOISE = 0.6
DEFAULT_HIRES_CFG = 4.0
DEFAULT_HIRES_SCALE = 1.25
HIRES_SCALE_MIN = 1.0
HIRES_SCALE_MAX = 4.0
HIRES_SCALE_STEP = 0.05
UPSCALE_LATENT_CUSTOM_MODE = "custom (sdxl_resizer)"
UPSCALE_LATENT_MODES = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp", UPSCALE_LATENT_CUSTOM_MODE]
DEFAULT_UPSCALE_LATENT_MODE = "bislerp"

# Image upscale defaults and limits
IMAGE_RESIZE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
DEFAULT_IMAGE_RESIZE_METHOD = "lanczos"
DEFAULT_IMAGE_UPSCALE_RESIZE = 0.5
IMAGE_UPSCALE_RESIZE_MIN = 0.05
IMAGE_UPSCALE_RESIZE_MAX = 4.0
IMAGE_UPSCALE_RESIZE_STEP = 0.05

# Image to latent (img2img source)
IMAGE_TO_LATENT_MODE_COVER = "cover (crop to fit)"
IMAGE_TO_LATENT_MODE_STRETCH = "stretch"
IMAGE_TO_LATENT_MODE_SOURCE = "source size"
IMAGE_TO_LATENT_MODES = [IMAGE_TO_LATENT_MODE_COVER, IMAGE_TO_LATENT_MODE_STRETCH, IMAGE_TO_LATENT_MODE_SOURCE]
DEFAULT_IMAGE_TO_LATENT_MODE = IMAGE_TO_LATENT_MODE_COVER

# Detailer (re-sample a masked region at a resolution the model can actually draw at)
DEFAULT_DETAILER_STEPS = 40
DEFAULT_DETAILER_SAMPLER = "deis"
DEFAULT_DETAILER_SCHEDULER = "beta"
DEFAULT_DETAILER_CFG = 4.0
DEFAULT_DETAILER_DENOISE = 0.5
# The size the *region* is scaled to, not the crop: a hand is what needs the pixels, and the context
# around it only has to be enough for the model to place the hand in.
DEFAULT_DETAILER_REGION_SIZE = 512
DETAILER_REGION_SIZE_MIN = 128
DETAILER_REGION_SIZE_MAX = 2048
DETAILER_REGION_SIZE_STEP = 64
# The ceiling the crop is not allowed to pass, which is what keeps the region size from turning a speck
# into a 4096 sample.
DEFAULT_DETAILER_MAX_SIZE = 1024
DETAILER_MAX_SIZE_MIN = 256
DETAILER_MAX_SIZE_MAX = 2048
DETAILER_MAX_SIZE_STEP = 64
# Context as a multiple of the region, not as a pixel count: a fixed padding gives the least context to
# the largest regions and starves the smallest ones of resolution.
DEFAULT_DETAILER_CROP_FACTOR = 3.0
DETAILER_CROP_FACTOR_MIN = 1.0
DETAILER_CROP_FACTOR_MAX = 8.0
DETAILER_CROP_FACTOR_STEP = 0.1
# How far the context may shrink below crop_factor so the region still reaches region_size under max_size.
# Past this the region gives up resolution instead: a hand needs its arm around it to be placed at all.
DETAILER_MIN_CROP_FACTOR = 1.5
DEFAULT_DETAILER_FEATHER = 32
DEFAULT_DETAILER_NOISE_FEATHER = 16
DETAILER_FEATHER_MIN = 0
DETAILER_FEATHER_MAX = 128
DETAILER_FEATHER_STEP = 1
DEFAULT_DETAILER_DIFFERENTIAL = True
DETAILER_MASK_THRESHOLD = 0.5
# A feathered mask's interior is snapped to 1.0 from this close: differential diffusion frees a cell on the
# first step only at exactly 1.0, and a blur leaves the interior a rounding error short of it.
DETAILER_MASK_SOLID_EPSILON = 1e-4
# A region section holding nothing but this takes that region out of the pass: `2: -`.
DETAILER_REGION_SKIP = "-"
DETAILER_MAX_REGIONS = 8
DETAILER_MIN_REGION_PIXELS = 64
DEFAULT_DETAILER_REGION_POSITIVE = "5 fingers"
DEFAULT_DETAILER_REGION_NEGATIVE = ""
# Where the detailer takes its seed from. The pipeline seed keeps a detail pass tied to the image it fixes;
# a fixed one lets the same regions be re-rolled while everything ahead of them stays put.
DETAILER_SEED_MODE_CONFIG = "from config"
DETAILER_SEED_MODE_FIXED = "fixed"
DETAILER_SEED_MODE_OFFSET = "config + offset"
DETAILER_SEED_MODES = [DETAILER_SEED_MODE_CONFIG, DETAILER_SEED_MODE_FIXED, DETAILER_SEED_MODE_OFFSET]
DEFAULT_DETAILER_SEED_MODE = DETAILER_SEED_MODE_CONFIG
DEFAULT_DETAILER_VARY_SEED = True
# Passes over the same crop, each starting from the last. One is a single pass: past ~0.5 denoise one pass breaks
# the hand's form, so a larger change has to come from several smaller ones.
DEFAULT_DETAILER_CYCLES = 1
DETAILER_CYCLES_MIN = 1
DETAILER_CYCLES_MAX = 4

# Detection (YOLO exported to ONNX, run through onnxruntime on the GPU)
DETECT_MODELS_RELATIVE_PATH = ("model", "detect")
DETECT_MODEL_EXTENSION = ".onnx"
DETECT_INPUT_SIZE = 640
# Tried in order; the CPU one is the fallback on a machine without usable CUDA.
DETECT_CUDA_PROVIDER = "CUDAExecutionProvider"
DETECT_EXECUTION_PROVIDERS = (DETECT_CUDA_PROVIDER, "CPUExecutionProvider")
# onnxruntime's severity scale: 3 is errors only.
DETECT_LOG_SEVERITY = 3
DETECT_SPINNING_OPTIONS = ("session.intra_op.allow_spinning", "session.inter_op.allow_spinning")
DETECT_PADDING_COLOR = 114
DEFAULT_DETECT_CONFIDENCE = 0.45
DETECT_CONFIDENCE_MIN = 0.05
DETECT_CONFIDENCE_MAX = 1.0
DETECT_CONFIDENCE_STEP = 0.05
DEFAULT_DETECT_IOU = 0.45
DETECT_IOU_MIN = 0.1
DETECT_IOU_MAX = 1.0
DETECT_IOU_STEP = 0.05
# Percent of the box's short side, not pixels: a fixed 16px grew a 40px hand by 80% and a 400px one by 8%,
# and the box is what sets the detailer's scale.
DEFAULT_DETECT_DILATION = 5
DETECT_DILATION_MIN = -50
DETECT_DILATION_MAX = 100
DETECT_DILATION_STEP = 1
DETECT_INFO_SCORES = 8
# The mask a detection is written as. `ellipse` keeps the box but eases its corners, which are almost always
# background, down to DETECT_ELLIPSE_CORNER: under differential diffusion that is a late start there rather
# than a blend, so the background in the corners survives and the region itself is still re-drawn in full.
# `segment` is the object's own outline, from a segmentation model; one without masks falls back to boxes.
DETECT_SHAPE_BOX = "box"
DETECT_SHAPE_ELLIPSE = "ellipse"
DETECT_SHAPE_SEGMENT = "segment"
DETECT_SHAPES = [DETECT_SHAPE_BOX, DETECT_SHAPE_ELLIPSE, DETECT_SHAPE_SEGMENT]
DEFAULT_DETECT_SHAPE = DETECT_SHAPE_SEGMENT
DETECT_ELLIPSE_CORNER = 0.3

# Box-prompted segmentation (MobileSAM as ONNX), the outline for detections that come without one
SEGMENT_MODELS_RELATIVE_PATH = ("model", "segment")
SEGMENT_ENCODER_FILE = "mobile_sam_image_encoder.onnx"
SEGMENT_DECODER_FILE = "sam_mask_decoder_single.onnx"
SEGMENT_INPUT_SIZE = 1024
# The crop the encoder sees, as a multiple of the box: enough context to tell the hand from what is behind
# it, small enough that the fingers still cover real cells of the embedding.
SEGMENT_CONTEXT_FACTOR = 2.0
DETECT_PREVIEW_COLOR = (255, 64, 64)
DETECT_PREVIEW_REFERENCE_SIZE = 1024
DETECT_PREVIEW_WIDTH = 3
DETECT_PREVIEW_FONT_SIZE = 28
DETECT_PREVIEW_FONT_MIN = 10

# Grid labels, tried in order. A CJK-capable face comes first because model names carry Chinese characters
# and Arial or DejaVu draws those as blanks; all of the leaders cover Latin, so plain labels look the same.
GRID_LABEL_FONTS = (
    "msyh.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "NotoSansCJK-Regular.ttc",
    "NotoSansCJKsc-Regular.otf",
    "simsun.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "arial.ttf",
)

# Video defaults
DEFAULT_VIDEO_FRAMES = 81
VIDEO_FRAMES_MIN = 17
VIDEO_FRAMES_MAX = 257
VIDEO_FRAMES_STEP = 16
DEFAULT_VIDEO_FPS = 16
DEFAULT_VIDEO_STEPS = 4
DEFAULT_VIDEO_CFG = 1.0
DEFAULT_VIDEO_SAMPLER = "euler"
DEFAULT_VIDEO_SCHEDULER = "simple"
VIDEO_FPS_MIN = 1
VIDEO_FPS_MAX = 60
VIDEO_FPS_STEP = 1
DEFAULT_WAN_SHIFT = 8.0
WAN_SHIFT_MIN = 0.0
WAN_SHIFT_MAX = 16.0
WAN_SHIFT_STEP = 0.5
# A sigma, not a fraction of the steps: the high-noise expert runs while the schedule sits at or above it.
# 0.9 is the value Wan 2.2 ships for I2V, and over four steps it happens to reproduce the old
# 0.5-of-steps split at any shift from 5 to 8. Outside that the two genuinely part ways, which is the point.
DEFAULT_WAN_BOUNDARY = 0.9
WAN_BOUNDARY_MIN = 0.0
WAN_BOUNDARY_MAX = 1.0
# Coarser than this is unusable: a high shift keeps the whole schedule above 0.7, so all of the useful
# range sits in the last few hundredths.
WAN_BOUNDARY_STEP = 0.01
DEFAULT_WAN_VACE_EMPTY_FRAME_LEVEL = 0.5
WAN_VACE_EMPTY_FRAME_LEVEL_MIN = 0.0
WAN_VACE_EMPTY_FRAME_LEVEL_MAX = 1.0
WAN_VACE_EMPTY_FRAME_LEVEL_STEP = 0.05
DEFAULT_WAN_VACE_STRENGTH = 1.0
WAN_VACE_STRENGTH_MIN = 0.0
WAN_VACE_STRENGTH_MAX = 1.0
WAN_VACE_STRENGTH_STEP = 0.05

# Video upscale defaults and limits
DEFAULT_VIDEO_UPSCALE_BATCH_SIZE = 16
VIDEO_TEMPORAL_SMOOTHING_MODES = ["off", "simple", "adaptive"]
DEFAULT_VIDEO_TEMPORAL_SMOOTHING_MODE = "adaptive"
DEFAULT_VIDEO_TEMPORAL_SMOOTHING = 0.12
DEFAULT_VIDEO_TEMPORAL_SHARPEN = 0.08
DEFAULT_VIDEO_MOTION_SENSITIVITY = 8.0
DEFAULT_VIDEO_EDGE_AWARE_SMOOTHING = True
DEFAULT_VIDEO_EDGE_PRESERVATION = 0.6
DEFAULT_VIDEO_SHARPEN = 0.04
VIDEO_TEMPORAL_SMOOTHING_MIN = 0.0
VIDEO_TEMPORAL_SMOOTHING_MAX = 0.5
VIDEO_TEMPORAL_SMOOTHING_STEP = 0.01
VIDEO_MOTION_SENSITIVITY_MIN = 0.1
VIDEO_MOTION_SENSITIVITY_MAX = 32.0
VIDEO_MOTION_SENSITIVITY_STEP = 0.1
VIDEO_UPSCALE_BATCH_SIZE_MIN = 1
VIDEO_UPSCALE_BATCH_SIZE_MAX = 128
VIDEO_UPSCALE_BATCH_SIZE_STEP = 1
VIDEO_SHARPEN_MIN = 0.0
VIDEO_SHARPEN_MAX = 0.5
VIDEO_SHARPEN_STEP = 0.01
VIDEO_EDGE_PRESERVATION_MIN = 0.0
VIDEO_EDGE_PRESERVATION_MAX = 1.0
VIDEO_EDGE_PRESERVATION_STEP = 0.01
VIDEO_TEMPORAL_SHARPEN_MIN = 0.0
VIDEO_TEMPORAL_SHARPEN_MAX = 0.5
VIDEO_TEMPORAL_SHARPEN_STEP = 0.01

# Image defaults
DEFAULT_IMAGE_WIDTH = 1024
DEFAULT_IMAGE_HEIGHT = 1024
IMAGE_SIZE_MIN = 64
IMAGE_SIZE_MAX = 4096
IMAGE_CHANNEL_MIN = 0
IMAGE_CHANNEL_MAX = 255
DEFAULT_IMAGE_CHANNEL = 255

# Image noise defaults
POWER_NOISE_TYPES = ["grey", "white", "pink", "blue", "green", "mix"]
DEFAULT_POWER_NOISE_TYPE = "white"
DEFAULT_POWER_NOISE_ATTENUATION = 0.5
# Above zero: at zero the grey noise every spectral type is filtered from has no deviation to normalise by.
POWER_NOISE_ATTENUATION_MIN = 0.01
POWER_NOISE_ATTENUATION_MAX = 10.0
POWER_NOISE_ATTENUATION_STEP = 0.01
POWER_NOISE_MAX_SEED = 4294967294
# Green noise is a band of radial frequency, in cycles per pixel (0 to ~0.71 on the diagonal).
POWER_NOISE_GREEN_CENTER = 0.25
POWER_NOISE_GREEN_WIDTH = 0.1

# Image filters
PIXEL_INIT_NONE = "none"
PIXEL_INIT_MODES = ["k-means++", "random", PIXEL_INIT_NONE]
DITHER_MODE_FLOYD = "floyd"
DITHER_MODE_ORDERED = "ordered"
DITHER_MODES = [DITHER_MODE_FLOYD, DITHER_MODE_ORDERED]
CANNY_MODES = ["normal", "laplacian"]
STYLE_FILTERS = [
    "1977",
    "aden",
    "brannan",
    "brooklyn",
    "clarendon",
    "earlybird",
    "fairy tale",
    "gingham",
    "hudson",
    "inkwell",
    "kelvin",
    "lark",
    "lofi",
    "maven",
    "mayfair",
    "moon",
    "nashville",
    "perpetua",
    "reyes",
    "rise",
    "slumber",
    "stinson",
    "toaster",
    "valencia",
    "walden",
    "willow",
    "xpro2",
]
BLEND_MODE_ADD = "add"
BLEND_MODES = [
    BLEND_MODE_ADD,
    "color",
    "color_burn",
    "color_dodge",
    "darken",
    "difference",
    "exclusion",
    "hard_light",
    "hue",
    "lighten",
    "multiply",
    "overlay",
    "screen",
    "soft_light",
]
DEFAULT_PIXEL_INIT_MODE = "random"
DEFAULT_DITHER_MODE = DITHER_MODE_FLOYD
DEFAULT_STYLE_FILTER = "fairy tale"
DEFAULT_CANNY_MODE = "normal"
DEFAULT_BLEND_MODE = "color"
DEFAULT_PIXELATE_SIZE = 128
PIXELATE_SIZE_MIN = 16
PIXELATE_SIZE_MAX = 480
DEFAULT_PIXELATE_COLORS = 32
PIXELATE_COLORS_MIN = 2
PIXELATE_COLORS_MAX = 256
DEFAULT_PIXELATE_DITHER = False
ADJUST_STEP = 0.01
DEFAULT_ADJUST_BRIGHTNESS = 0.0
ADJUST_BRIGHTNESS_MIN = -1.0
ADJUST_BRIGHTNESS_MAX = 1.0
DEFAULT_ADJUST_CONTRAST = 1.0
ADJUST_CONTRAST_MIN = -1.0
ADJUST_CONTRAST_MAX = 2.0
DEFAULT_ADJUST_SATURATION = 1.0
ADJUST_SATURATION_MIN = 0.0
ADJUST_SATURATION_MAX = 5.0
DEFAULT_ADJUST_SHARPNESS = 1.0
ADJUST_SHARPNESS_MIN = -5.0
ADJUST_SHARPNESS_MAX = 5.0
DEFAULT_ADJUST_BLUR = 0
ADJUST_BLUR_MIN = 0
ADJUST_BLUR_MAX = 16
DEFAULT_ADJUST_GAUSSIAN_BLUR = 0.0
ADJUST_GAUSSIAN_BLUR_MIN = 0.0
ADJUST_GAUSSIAN_BLUR_MAX = 1024.0
ADJUST_GAUSSIAN_BLUR_STEP = 0.1
DEFAULT_ADJUST_EDGE_ENHANCE = 0.0
ADJUST_EDGE_ENHANCE_MIN = 0.0
ADJUST_EDGE_ENHANCE_MAX = 1.0
DEFAULT_ADJUST_DETAIL_ENHANCE = True
DEFAULT_LUCY_ITERATIONS = 12
LUCY_ITERATIONS_MIN = 1
LUCY_ITERATIONS_MAX = 16
DEFAULT_LUCY_KERNEL_SIZE = 3
LUCY_KERNEL_SIZE_MIN = 1
LUCY_KERNEL_SIZE_MAX = 16
DEFAULT_HIGH_PASS_RADIUS = 10
HIGH_PASS_RADIUS_MIN = 1
HIGH_PASS_RADIUS_MAX = 500
DEFAULT_HIGH_PASS_STRENGTH = 1.5
HIGH_PASS_STRENGTH_MIN = 0.0
HIGH_PASS_STRENGTH_MAX = 255.0
HIGH_PASS_STRENGTH_STEP = 0.1
DEFAULT_HIGH_PASS_COLOR_OUTPUT = True
DEFAULT_HIGH_PASS_NEUTRAL_BACKGROUND = True
# Mid-grey, the level a neutral high pass centres the detail on.
HIGH_PASS_NEUTRAL_LEVEL = 128.0
DEFAULT_LEVELS_BLACK = 0.0
DEFAULT_LEVELS_MID = 127.5
DEFAULT_LEVELS_WHITE = 255.0
LEVELS_MIN = 0.0
LEVELS_MAX = 255.0
LEVELS_STEP = 0.1
DRAGON_MIN = 0.0
DRAGON_MAX = 16.0
DRAGON_STEP = 0.01
DEFAULT_DRAGON_SATURATION = 1.0
DEFAULT_DRAGON_CONTRAST = 1.0
DEFAULT_DRAGON_BRIGHTNESS = 1.0
DEFAULT_DRAGON_SHARPNESS = 1.0
DRAGON_SHARPNESS_MAX = 6.0
DEFAULT_DRAGON_HIGHPASS_SAMPLES = 1
DRAGON_HIGHPASS_SAMPLES_MIN = 0
DRAGON_HIGHPASS_SAMPLES_MAX = 6
DEFAULT_DRAGON_HIGHPASS_STRENGTH = 1.0
DRAGON_HIGHPASS_STRENGTH_MAX = 3.0
DEFAULT_DRAGON_HIGHPASS_RADIUS = 6.0
DRAGON_HIGHPASS_RADIUS_MAX = 255.0
DEFAULT_DRAGON_COLORIZE = True
OPACITY_MIN = 0.0
OPACITY_MAX = 1.0
OPACITY_STEP = 0.01
DEFAULT_BLEND_MODE_OPACITY = 1.0
DEFAULT_BLEND_MASK_OPACITY = 0.5
DEFAULT_MASK_BLUR_KERNEL_SIZE = 10
MASK_BLUR_KERNEL_SIZE_MIN = 0
MASK_BLUR_KERNEL_SIZE_MAX = 100
DEFAULT_MASK_BLUR_SIGMA = 10.0
MASK_BLUR_SIGMA_MIN = 0.1
MASK_BLUR_SIGMA_MAX = 100.0
MASK_BLUR_SIGMA_STEP = 0.1

# Latent upscale (sdxl_resizer)
DEFAULT_LATENT_SCALE_BY = 1.5
LATENT_SCALE_BY_MIN = 1.0
LATENT_SCALE_BY_MAX = 4.0
LATENT_SCALE_BY_STEP = 0.25

# XY / XYZ grids
GRID_THEMES = ("light", "dark")
DEFAULT_GRID_THEME = "dark"
DEFAULT_GRID_SHOW_PARTIAL = True
DEFAULT_GRID_CLEAR_ON_COMPLETE = True
DEFAULT_GRID_CLEAR_ON_FIRST_CELL = True
DEFAULT_GRID_EMPTY_CELLS = True
DEFAULT_GRID_CELL_GAP = 12
DEFAULT_GRID_PADDING = 16
GRID_GAP_MIN = 0
GRID_GAP_MAX = 256
DEFAULT_GRID_Z_GAP = 24
GRID_Z_GAP_MAX = 512
DEFAULT_GRID_LABEL_SIZE = 24
GRID_LABEL_SIZE_MIN = 8
GRID_LABEL_SIZE_MAX = 128

# Remove background post-processing
DEFAULT_REMBG_SHARPEN_ITERATIONS = 14
DEFAULT_REMBG_SHARPEN_KERNEL = 3
REMBG_SHARPEN_MIN = 1
REMBG_SHARPEN_MAX = 16
DEFAULT_REMBG_CONTRAST = 2.0
REMBG_CONTRAST_MIN = -1.0
REMBG_CONTRAST_MAX = 2.0
REMBG_CONTRAST_STEP = 0.01
DEFAULT_REMBG_BLUR = 2
REMBG_BLUR_MIN = 0
REMBG_BLUR_MAX = 16

# Lists
DEFAULT_LIST_INDEX = 1
LIST_INDEX_MIN = 1
LIST_INDEX_MAX = 99

# Image save defaults and limits
DEFAULT_SAVE_DATE_FORMAT = "%Y-%m-%d"
DEFAULT_SAVE_IMAGE_FOLDER = "img/[date]"
DEFAULT_SAVE_IMAGE_PREFIX = "img"
DEFAULT_SAVE_DELIMITER = "-"
DEFAULT_SAVE_EXTENSION = "png"
DEFAULT_SAVE_DPI = 300
SAVE_DPI_MIN = 1
SAVE_DPI_MAX = 2400
DEFAULT_SAVE_QUALITY = 100
SAVE_QUALITY_MIN = 1
SAVE_QUALITY_MAX = 100
DEFAULT_SAVE_NUMBER_PADDING = 4
SAVE_NUMBER_PADDING_MIN = 1
SAVE_NUMBER_PADDING_MAX = 9

# Prompt defaults
PROMPT_SEPARATOR_SEMICOLON = "semicolon [;]"
PROMPT_SEPARATORS = ["break", "comma [,]", PROMPT_SEPARATOR_SEMICOLON]
DEFAULT_PROMPT_SEPARATOR = "comma [,]"
DEFAULT_PROCESS_TAG_WEIGHTS = True
POSITIVE_STYLE_POSITIONS = ["middle", "start", "end"]
DEFAULT_POSITIVE_STYLE_POSITION = "end"
DEFAULT_POSITIVE_PROMPT = ""
DEFAULT_POSITIVE_STYLE = (
    "masterpiece, best quality, amazing quality, very aesthetic, absurdres, highly detailed, "
    "beautiful detailed eyes, detailed face, detailed skin, detailed hair, detailed hands, perfect anatomy, sharp focus"
)
DEFAULT_NEGATIVE_PROMPT = ""
DEFAULT_NEGATIVE_STYLE = (
    "worst quality, low quality, blurry, bad anatomy, bad hands, deformed hands, "
    "extra fingers, fused fingers, missing fingers, long fingers, extra limbs, watermark, signature, text, logo"
)

# Lora defaults and limits
DEFAULT_LORA_STRENGTH = 1.0
LORA_STRENGTH_MIN = -10.0
LORA_STRENGTH_MAX = 10.0
LORA_STRENGTH_STEP = 0.05

# Math options
MATH_ACTIONS = ["a + b", "a - b", "a * b", "a / b"]
COMPARE_ACTIONS = ["a < b", "a > b", "a == b", "a <= b", "a >= b"]

# Runtime assets
AUTOCOMPLETE_RELATIVE_PATH = ("autocomplete", "autocomplete.txt")
LATENT_RESIZER_MODEL_RELATIVE_PATH = ("model", "sdxl_resizer.pt")
LATENT_RESIZER_SCALE_FACTOR = 0.13025
BIREFNET_MODEL_RELATIVE_PATH = ("model", "rembg", "birefnet.safetensors")
BIREFNET_INPUT_SIZE = (1024, 1024)
# Activations of one 1024x1024 forward, asked of the model manager on top of the weights (2.3 GB measured).
BIREFNET_INFERENCE_MEMORY = 3 * 1024 ** 3
