import numpy as np


def parse_keypoints(lankmarks, idxs: list = None, threshold: float = 0.0) -> np.ndarray:
    if lankmarks is None:
        return np.zeros((len(idxs), 3))

    idxs = range(len(lankmarks.landmark)) if idxs is None else idxs
    keypoints = np.zeros((len(idxs), 3))
    for i, idx in enumerate(idxs):
        keypoint = lankmarks.landmark[idx]
        if keypoint.visibility >= threshold:
            keypoints[i] = [keypoint.x, keypoint.y, keypoint.z]

    return keypoints
