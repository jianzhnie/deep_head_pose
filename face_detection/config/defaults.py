import torch
from yacs.config import CfgNode as CN

# ----------------------------------------------------------------------------
# Config definition
# ----------------------------------------------------------------------------

_C = CN()

# ----------------------------------------------------------------------------
# DETECTION
# ----------------------------------------------------------------------------
_C.DETECTION = CN()

_C.DETECTION.CUDA_DEVICE = "cuda:0"
_C.DETECTION.THRESHOLD = 0.30
_C.DETECTION.INPUT_SIZE = (864, 486)

# ----------------------------------------------------------------------------
# DRAWING
# ----------------------------------------------------------------------------
_C.DETECTION.DRAWING = CN()
# The color to draw bboxes (in RGB)
_C.DETECTION.DRAWING.BBOX_COLOR = (0, 255, 0)
# The line width of bboxes
_C.DETECTION.DRAWING.BBOX_WIDTH = 2

# ----------------------------------------------------------------------------
# origin configs:
# data augument config
_C.expand_prob = 0.5
_C.expand_max_ratio = 4
_C.hue_prob = 0.5
_C.hue_delta = 18
_C.contrast_prob = 0.5
_C.contrast_delta = 0.5
_C.saturation_prob = 0.5
_C.saturation_delta = 0.5
_C.brightness_prob = 0.5
_C.brightness_delta = 0.125
_C.data_anchor_sampling_prob = 0.5
_C.min_face_size = 6.0
_C.apply_distort = True
_C.apply_expand = False
_C.img_mean = [104., 117., 123.]
_C.resize_width = 640
_C.resize_height = 640
_C.scale = 1 / 127.0
_C.anchor_sampling = True
_C.filter_min_face = True

# train config
_C.LR_STEPS = (80000, 100000, 120000)
_C.MAX_STEPS = 150000
_C.EPOCHES = 100

# anchor config
_C.FEATURE_MAPS = [160, 80, 40, 20, 10, 5]
_C.INPUT_SIZE = 640
_C.STEPS = [4, 8, 16, 32, 64, 128]
_C.ANCHOR_SIZES1 = [8, 16, 32, 64, 128, 256]
_C.ANCHOR_SIZES2 = [16, 32, 64, 128, 256, 512]
_C.ASPECT_RATIO = [1.0]
_C.CLIP = False
_C.VARIANCE = [0.1, 0.2]

# detection config
_C.NMS_THRESH = 0.3
_C.NMS_TOP_K = 5000
_C.TOP_K = 20
_C.CONF_THRESH = 0.01

# loss config
_C.NEG_POS_RATIOS = 3
_C.NUM_CLASSES = 2
