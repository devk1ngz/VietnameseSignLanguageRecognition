import random
from pose_format import Pose


class SLGCNAugment:
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
            selected_aug = random.randrange(3)
            if selected_aug == 0:
                return pose.augment2d(rotation_std=self.rotation_std)
            if selected_aug == 1:
                return pose.augment2d(shear_std=self.shear_std)
            if selected_aug == 2:
                return pose.augment2d(scale_std=self.scale_std)
        return pose
