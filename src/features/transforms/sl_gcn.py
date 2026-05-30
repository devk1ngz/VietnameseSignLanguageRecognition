import torch
import numpy as np
from pose_format import Pose
from utils import SLGCN_JOINTS, COCO_TO_POSE_FORMAT


class SLGCNJointSelect:
    def __init__(self, num_points: int = 27) -> None:
        self.joints = SLGCN_JOINTS[num_points]

    def __get_point(self, component: str, point: str, pose: Pose) -> np.ndarray:
        idx = pose.header._get_point_index(component, point)
        T, _, _, C = pose.body.data.shape
        data = np.zeros((T, C), dtype=pose.body.data.dtype)
        data[:, :2] = pose.body.data[:, 0, idx, :2].data
        data[:, 2] = pose.body.confidence[:, 0, idx]
        return data

    def __call__(self, pose: Pose) -> np.ndarray:
        pose.normalize_distribution()
        data = []
        for joint in self.joints:
            component, point = COCO_TO_POSE_FORMAT[joint]
            data.append(self.__get_point(component, point, pose))
        # (num_landmarks, num_frames, 3) -> (num_frames, num_landmarks, 3)
        return np.array(data).transpose((1, 0, 2))


class SLGCNPad:
    def __init__(self, num_frames: int = 150) -> None:
        self.num_frames = num_frames

    def __call__(self, data: np.ndarray) -> np.ndarray:
        padded_data = np.zeros(
            (self.num_frames, data.shape[1], data.shape[2], 1),
            dtype=np.float32,
        )
        L = data.shape[0]
        if L < self.num_frames:
            padded_data[:L, :, :, 0] = data
            rest = self.num_frames - L
            num = int(np.ceil(rest / L))
            pad = np.concatenate([data for _ in range(num)], 0)[:rest]
            padded_data[L:, :, :, 0] = pad
        else:
            padded_data[:, :, :, 0] = data[:self.num_frames, :, :]
        # (num_frames, num_points, num_channels, num_people)
        # -> (num_channels, num_frames, num_points, num_people)
        padded_data = np.transpose(padded_data, [2, 0, 1, 3])
        return padded_data


class SLGCNMotionStream:
    def __call__(self, data: np.ndarray) -> np.ndarray:
        T = data.shape[1]
        ori_data = data
        for t in range(T - 1):
            data[:, t, :, :] = ori_data[:, t + 1, :, :] - ori_data[:, t, :, :]
        data[:, T - 1, :, :] = 0
        return data


class SLGCNBoneStream:
    def __init__(self) -> None:
        self.ori_idxs = (
            (5, 6),
            (5, 7),
            (6, 8),
            (8, 10),
            (7, 9),
            (9, 11),
            (12, 13),
            (12, 14),
            (12, 16),
            (12, 18),
            (12, 20),
            (14, 15),
            (16, 17),
            (18, 19),
            (20, 21),
            (22, 23),
            (22, 24),
            (22, 26),
            (22, 28),
            (22, 30),
            (24, 25),
            (26, 27),
            (28, 29),
            (30, 31),
            (10, 12),
            (11, 22),
        )

    def __call__(self, data: np.ndarray) -> np.ndarray:
        ori_data = data
        for v1, v2 in self.ori_idxs:
            data[:, :, v2 - 5, :] = (
                ori_data[:, :, v2 - 5, :] - ori_data[:, :, v1 - 5, :]
            )
        return data


class NumPyToTensor:
    def __call__(self, data: np.ndarray) -> torch.Tensor:
        """
        Converts a numpy array to a PyTorch tensor.
        """
        return torch.from_numpy(data)


class SLGCNNormalize:
    def __init__(self, is_vector: bool = False):
        self.is_vector = is_vector

    def __call__(self, data: np.ndarray) -> np.ndarray:
        assert data.shape[0] == 3
        if self.is_vector:
            data[0, :, 0, :] = data[0, :, 0, :] - data[0, :, 0, 0].mean(axis=0)
            data[1, :, 0, :] = data[1, :, 0, :] - data[1, :, 0, 0].mean(axis=0)
        else:
            data[0, :, :, :] = data[0, :, :, :] - data[0, :, 0, 0].mean(axis=0)
            data[1, :, :, :] = data[1, :, :, :] - data[1, :, 0, 0].mean(axis=0)
        return data
