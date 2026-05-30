import torch
import numpy as np
import onnxruntime as ort
from typing import Dict, Any, Union
from transformers import Pipeline, AutoConfig
from huggingface_hub import hf_hub_download
from torchvision.transforms.v2 import Compose
from pathlib import Path
from pose_format import Pose
from pose_format.utils.holistic import load_holistic


JOINTS = {
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


class SLGCNGraphClassificationPipeline(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if kwargs.pop("use_onnx", False):
            repo_id = self.model.config._name_or_path
            model_kwargs = kwargs.get("model_kwargs", {})
            model_file = hf_hub_download(
                repo_id=repo_id,
                filename=f"{repo_id.split('/')[1]}.onnx",
                cache_dir=model_kwargs.get("cache_dir", "models/huggingface"),
            )
            self.config = AutoConfig.from_pretrained(
                repo_id,
                trust_remote_code=True,
                cache_dir=model_kwargs.get("cache_dir", "models/huggingface"),
            )
            self.id2label = self.config.id2label
            self.model = ort.InferenceSession(model_file)
        else:
            self.id2label = self.model.config.id2label

        self.transforms = [
            PoseExtract(),
            JointSelect(self.feature_extractor.num_points),
            Pad(self.feature_extractor.num_frames),
        ]
        if kwargs.get("bone_stream", False):
            self.transforms.append(BoneStream())
        if kwargs.get("motion_stream", False):
            self.transforms.append(MotionStream())
        self.transforms.extend(
            [
                Normalize(self.feature_extractor.is_vector),
                NumPyToTensor(),
            ]
        )
        self.transforms = Compose(self.transforms)

    def _sanitize_parameters(self, **kwargs):
        # Sanitize the parameters for preprocessing
        preprocess_kwargs = {}
        # Sanitize the parameters for the forward pass
        forward_kwargs = {}
        # Sanitize the parameters for postprocessing
        postprocess_kwargs = {}
        postprocess_kwargs["top_k"] = kwargs.get("top_k", 3)

        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, inputs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Preprocesses the inputs to the model.

        Parameters
        ----------
        inputs : Union[np.ndarray, torch.Tensor]
            The inputs to the model (time, height, width, channels).

        Returns
        -------
        torch.Tensor
            The preprocessed inputs (batch, channels, time, height, width).
        """
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs)
        return self.transforms(inputs).unsqueeze(0)

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if isinstance(self.model, ort.InferenceSession):
            inputs = inputs.cpu().numpy()
            return torch.from_numpy(self.model.run(None, {"poses": inputs})[0])
        return self.model(inputs.to(self.device)).logits

    def postprocess(self, logits: torch.Tensor, top_k: int = 3) -> list:
        logits = logits.cpu()

        topk_scores, topk_indices = torch.topk(logits, top_k, dim=1)
        topk_scores = torch.nn.functional.softmax(topk_scores, dim=1)
        topk_scores = topk_scores.squeeze().detach().numpy()
        topk_indices = topk_indices.squeeze().detach().numpy()

        return [
            {
                'gloss': self.id2label[str(topk_indices[i])],
                'score': topk_scores[i],
            }
            for i in range(top_k)
        ]


class PoseExtract:
    def __call__(self, inputs: Union[Dict[str, Any], str, Path]) -> Pose:
        pose = load_holistic(
            frames=inputs["frames"],
            fps=inputs["fps"],
            width=inputs["width"],
            height=inputs["height"],
            progress=False,
        )
        pose.normalize_distribution()
        return pose


class JointSelect:
    def __init__(self, num_points: int = 27) -> None:
        self.joints = JOINTS[num_points]

    def __get_point(self, component: str, point: str, pose: Pose) -> np.ndarray:
        idx = pose.header._get_point_index(component, point)
        T, _, _, C = pose.body.data.shape
        data = np.zeros((T, C), dtype=pose.body.data.dtype)
        data[:, :2] = pose.body.data[:, 0, idx, :2].data
        data[:, 2] = pose.body.confidence[:, 0, idx]
        return data

    def __call__(self, pose: Pose) -> np.ndarray:
        data = []
        for joint in self.joints:
            component, point = COCO_TO_POSE_FORMAT[joint]
            data.append(self.__get_point(component, point, pose))
        # (num_landmarks, num_frames, 3) -> (num_frames, num_landmarks, 3)
        return np.array(data).transpose((1, 0, 2))


class Pad:
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


class MotionStream:
    def __call__(self, data: np.ndarray) -> np.ndarray:
        T = data.shape[1]
        ori_data = data
        for t in range(T - 1):
            data[:, t, :, :] = ori_data[:, t + 1, :, :] - ori_data[:, t, :, :]
        data[:, T - 1, :, :] = 0
        return data


class BoneStream:
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


class Normalize:
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
