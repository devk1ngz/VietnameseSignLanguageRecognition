import random
import numpy as np
from pose_format import Pose


class SLGCNAugment:
    """
    Geometric augmentation in the pose (2D) domain. With probability
    ``aug_prob`` a random rotation, shear and scale are applied *together*
    (sampled from per-transform Gaussians), which is a stronger and more
    diverse augmentation than applying a single transform at a time.
    """

    def __init__(
        self,
        aug_prob: float = 0.5,
        rotation_std: float = 0.2,
        shear_std: float = 0.2,
        scale_std: float = 0.2,
    ) -> None:
        self.aug_prob = aug_prob
        self.rotation_std = rotation_std
        self.shear_std = shear_std
        self.scale_std = scale_std

    def __call__(self, pose: Pose) -> Pose:
        if random.random() < self.aug_prob:
            return pose.augment2d(
                rotation_std=self.rotation_std,
                shear_std=self.shear_std,
                scale_std=self.scale_std,
            )
        return pose


class SLGCNRandomMasking:
    """
    Coordinate-space regularisation applied at training time (operates on the
    padded array of shape (C, T, V, M)):
      - joint masking: randomly zero out whole joints across all frames,
        simulating missed/occluded keypoints (which MediaPipe produces often).
      - frame masking: randomly zero out whole frames, encouraging temporal
        robustness.
    """

    def __init__(
        self,
        joint_mask_prob: float = 0.0,
        frame_mask_prob: float = 0.0,
    ) -> None:
        self.joint_mask_prob = joint_mask_prob
        self.frame_mask_prob = frame_mask_prob

    def __call__(self, data: np.ndarray) -> np.ndarray:
        C, T, V, M = data.shape
        if self.joint_mask_prob > 0:
            joint_mask = np.random.rand(V) < self.joint_mask_prob
            if joint_mask.any():
                data[:, :, joint_mask, :] = 0.0
        if self.frame_mask_prob > 0:
            frame_mask = np.random.rand(T) < self.frame_mask_prob
            if frame_mask.any():
                data[:, frame_mask, :, :] = 0.0
        return data
