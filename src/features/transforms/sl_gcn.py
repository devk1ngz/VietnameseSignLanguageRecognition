import torch
import numpy as np
from pose_format import Pose
from utils import SLGCN_JOINTS, COCO_TO_POSE_FORMAT
from models.sl_gcn.modelling import _build_edges


def _build_bone_pairs(num_points: int) -> list:
    """
    Return a list of (child, parent) joint index pairs describing the skeleton
    tree, derived from the same anatomical edges used by the model graph.
    The tree is rooted at joint 0 via BFS; the root has no bone.
    """
    edges = _build_edges(num_points)
    neighbours = {v: [] for v in range(num_points)}
    for i, j in edges:
        neighbours[i].append(j)
        neighbours[j].append(i)
    parent = {0: None}
    queue = [0]
    while queue:
        node = queue.pop(0)
        for nxt in neighbours[node]:
            if nxt not in parent:
                parent[nxt] = node
                queue.append(nxt)
    return [(child, par) for child, par in parent.items() if par is not None]


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
    """Temporal motion stream: per-frame displacement (x[t+1] - x[t])."""

    def __call__(self, data: np.ndarray) -> np.ndarray:
        # data: (C, T, V, M)
        motion = np.zeros_like(data)
        motion[:, :-1, :, :] = data[:, 1:, :, :] - data[:, :-1, :, :]
        return motion


class SLGCNBoneStream:
    """
    Bone stream: each joint is replaced by the vector to its parent joint,
    using the anatomical skeleton tree (consistent with the model graph).
    The root joint has no parent and is left as zeros.
    """

    def __init__(self, num_points: int = 27) -> None:
        self.pairs = _build_bone_pairs(num_points)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        # data: (C, T, V, M)
        bone = np.zeros_like(data)
        for child, parent in self.pairs:
            bone[:, :, child, :] = data[:, :, child, :] - data[:, :, parent, :]
        return bone


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
