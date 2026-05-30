import numpy as np


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

TORCHHUB_RGB_BASED_MODELS = (
    'swin3d_t',
    'swin3d_s',
    'swin3d_b',
    "r3d_18",
    "mc3_18",
    "r2plus1d_18",
    "s3d",
    "mvit_v1_b",
    "mvit_v2_s",
)
HUGGINGFACE_RGB_BASED_MODELS = (
    "videomae",
)
RGB_BASED_MODELS = HUGGINGFACE_RGB_BASED_MODELS + TORCHHUB_RGB_BASED_MODELS

POSE_BASED_MODELS = (
    "spoter",
    "sl_gcn",
    "dsta_slr"
)

MODELS = RGB_BASED_MODELS + POSE_BASED_MODELS

HAND_LANDMARKS = [
    "wrist",
    "indexTip",
    "indexDIP",
    "indexPIP",
    "indexMCP",
    "middleTip",
    "middleDIP",
    "middlePIP",
    "middleMCP",
    "ringTip",
    "ringDIP",
    "ringPIP",
    "ringMCP",
    "littleTip",
    "littleDIP",
    "littlePIP",
    "littleMCP",
    "thumbTip",
    "thumbIP",
    "thumbMP",
    "thumbCMC",
]

BODY_LANDMARKS = [
    "nose",
    "neck",
    "rightEye",
    "leftEye",
    "rightEar",
    "leftEar",
    "rightShoulder",
    "leftShoulder",
    "rightElbow",
    "leftElbow",
    "rightWrist",
    "leftWrist",
]

ARM_LANDMARKS_ORDER = ["neck", "$side$Shoulder", "$side$Elbow", "$side$Wrist"]

HANDS_LANDMARKS = [
    id + suffix
    for id in HAND_LANDMARKS
    for suffix in ["_0", "_1"]
]

LANDMARKS = BODY_LANDMARKS + HANDS_LANDMARKS

SLGCN_JOINTS = {
    59: np.concatenate((np.arange(0, 17), np.arange(91, 133)), axis=0),  # 59
    31: np.concatenate(
        (
            np.arange(0, 11),
            [91, 95, 96, 99, 100, 103, 104, 107, 108, 111],
            [112, 116, 117, 120, 121, 124, 125, 128, 129, 132],
        ),
        axis=0,
    ),  # 31
    27: np.concatenate(
        (
            [0, 5, 6, 7, 8, 9, 10],
            [91, 95, 96, 99, 100, 103, 104, 107, 108, 111],
            [112, 116, 117, 120, 121, 124, 125, 128, 129, 132],
        ),
        axis=0,
    ),  # 27
}

COCO_TO_POSE_FORMAT = {
    0: ("POSE_LANDMARKS", "NOSE"),
    1: ("POSE_LANDMARKS", "LEFT_EYE"),
    2: ("POSE_LANDMARKS", "RIGHT_EYE"),
    3: ("POSE_LANDMARKS", "LEFT_EAR"),
    4: ("POSE_LANDMARKS", "RIGHT_EAR"),
    5: ("POSE_LANDMARKS", "LEFT_SHOULDER"),
    6: ("POSE_LANDMARKS", "RIGHT_SHOULDER"),
    7: ("POSE_LANDMARKS", "LEFT_ELBOW"),
    8: ("POSE_LANDMARKS", "RIGHT_ELBOW"),
    9: ("POSE_LANDMARKS", "LEFT_WRIST"),
    10: ("POSE_LANDMARKS", "RIGHT_WRIST"),
    11: ("POSE_LANDMARKS", "LEFT_HIP"),
    12: ("POSE_LANDMARKS", "RIGHT_HIP"),
    13: ("POSE_LANDMARKS", "LEFT_KNEE"),
    14: ("POSE_LANDMARKS", "RIGHT_KNEE"),
    15: ("POSE_LANDMARKS", "LEFT_ANKLE"),
    16: ("POSE_LANDMARKS", "RIGHT_ANKLE"),
    91: ("LEFT_HAND_LANDMARKS", "WRIST"),
    92: ("LEFT_HAND_LANDMARKS", "THUMB_CMC"),
    93: ("LEFT_HAND_LANDMARKS", "THUMB_MCP"),
    94: ("LEFT_HAND_LANDMARKS", "THUMB_IP"),
    95: ("LEFT_HAND_LANDMARKS", "THUMB_TIP"),
    96: ("LEFT_HAND_LANDMARKS", "INDEX_FINGER_MCP"),
    97: ("LEFT_HAND_LANDMARKS", "INDEX_FINGER_PIP"),
    98: ("LEFT_HAND_LANDMARKS", "INDEX_FINGER_DIP"),
    99: ("LEFT_HAND_LANDMARKS", "INDEX_FINGER_TIP"),
    100: ("LEFT_HAND_LANDMARKS", "MIDDLE_FINGER_MCP"),
    101: ("LEFT_HAND_LANDMARKS", "MIDDLE_FINGER_PIP"),
    102: ("LEFT_HAND_LANDMARKS", "MIDDLE_FINGER_DIP"),
    103: ("LEFT_HAND_LANDMARKS", "MIDDLE_FINGER_TIP"),
    104: ("LEFT_HAND_LANDMARKS", "RING_FINGER_MCP"),
    105: ("LEFT_HAND_LANDMARKS", "RING_FINGER_PIP"),
    106: ("LEFT_HAND_LANDMARKS", "RING_FINGER_DIP"),
    107: ("LEFT_HAND_LANDMARKS", "RING_FINGER_TIP"),
    108: ("LEFT_HAND_LANDMARKS", "PINKY_MCP"),
    109: ("LEFT_HAND_LANDMARKS", "PINKY_PIP"),
    110: ("LEFT_HAND_LANDMARKS", "PINKY_DIP"),
    111: ("LEFT_HAND_LANDMARKS", "PINKY_TIP"),
    112: ("RIGHT_HAND_LANDMARKS", "WRIST"),
    113: ("RIGHT_HAND_LANDMARKS", "THUMB_CMC"),
    114: ("RIGHT_HAND_LANDMARKS", "THUMB_MCP"),
    115: ("RIGHT_HAND_LANDMARKS", "THUMB_IP"),
    116: ("RIGHT_HAND_LANDMARKS", "THUMB_TIP"),
    117: ("RIGHT_HAND_LANDMARKS", "INDEX_FINGER_MCP"),
    118: ("RIGHT_HAND_LANDMARKS", "INDEX_FINGER_PIP"),
    119: ("RIGHT_HAND_LANDMARKS", "INDEX_FINGER_DIP"),
    120: ("RIGHT_HAND_LANDMARKS", "INDEX_FINGER_TIP"),
    121: ("RIGHT_HAND_LANDMARKS", "MIDDLE_FINGER_MCP"),
    122: ("RIGHT_HAND_LANDMARKS", "MIDDLE_FINGER_PIP"),
    123: ("RIGHT_HAND_LANDMARKS", "MIDDLE_FINGER_DIP"),
    124: ("RIGHT_HAND_LANDMARKS", "MIDDLE_FINGER_TIP"),
    125: ("RIGHT_HAND_LANDMARKS", "RING_FINGER_MCP"),
    126: ("RIGHT_HAND_LANDMARKS", "RING_FINGER_PIP"),
    127: ("RIGHT_HAND_LANDMARKS", "RING_FINGER_DIP"),
    128: ("RIGHT_HAND_LANDMARKS", "RING_FINGER_TIP"),
    129: ("RIGHT_HAND_LANDMARKS", "PINKY_MCP"),
    130: ("RIGHT_HAND_LANDMARKS", "PINKY_PIP"),
    131: ("RIGHT_HAND_LANDMARKS", "PINKY_DIP"),
    132: ("RIGHT_HAND_LANDMARKS", "PINKY_TIP"),
}
