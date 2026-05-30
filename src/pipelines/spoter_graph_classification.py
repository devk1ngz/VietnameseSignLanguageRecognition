import torch
import logging
import numpy as np
import onnxruntime as ort
from typing import Dict, Any, Union
from transformers import Pipeline, AutoConfig
from huggingface_hub import hf_hub_download
from torchvision.transforms.v2 import Compose
from pathlib import Path
from pose_format import Pose
from pose_format.utils.holistic import load_holistic


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


class SPOTERGraphClassificationPipeline(Pipeline):
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

        self.transforms = Compose(
            [
                PoseExtract(),
                JointSelect(),
                TensorToDict(),
                SingleBodyDictNormalize(),
                SPOTERSingleHandDictNormalize(),
                DictToTensor(),
                Shift()
            ]
        )

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
        return pose


class JointSelect:
    def __init__(self) -> None:
        self.pose_landmarks = [
            "NOSE",
            "NECK",
            "RIGHT_EYE",
            "LEFT_EYE",
            "RIGHT_EAR",
            "LEFT_EAR",
            "RIGHT_SHOULDER",
            "LEFT_SHOULDER",
            "RIGHT_ELBOW",
            "LEFT_ELBOW",
            "RIGHT_WRIST",
            "LEFT_WRIST",
        ]
        self.hand_landmarks = [
            "WRIST",
            "INDEX_FINGER_TIP",
            "INDEX_FINGER_DIP",
            "INDEX_FINGER_PIP",
            "INDEX_FINGER_MCP",
            "MIDDLE_FINGER_TIP",
            "MIDDLE_FINGER_DIP",
            "MIDDLE_FINGER_PIP",
            "MIDDLE_FINGER_MCP",
            "RING_FINGER_TIP",
            "RING_FINGER_DIP",
            "RING_FINGER_PIP",
            "RING_FINGER_MCP",
            "PINKY_DIP",
            "PINKY_TIP",
            "PINKY_PIP",
            "PINKY_MCP",
            "THUMB_TIP",
            "THUMB_IP",
            "THUMB_MCP",
            "THUMB_CMC",
        ]

    def __get_point(self, component: str, point: str, pose: Pose) -> np.ndarray:
        if point == "NECK":
            return np.zeros_like(pose.body.data[:, 0, 0, :2].data)
        idx = pose.header._get_point_index(component, point)
        return pose.body.data[:, 0, idx, :2].data

    def __call__(self, pose: Pose) -> torch.Tensor:
        data = []
        for point in self.pose_landmarks:
            data.append(self.__get_point("POSE_LANDMARKS", point, pose))
        for side in ["LEFT", "RIGHT"]:
            for point in self.hand_landmarks:
                data.append(self.__get_point(f"{side}_HAND_LANDMARKS", point, pose))
        # (num_landmarks, num_frames, 2) -> (num_frames, num_landmarks, 2)
        data = np.array(data).transpose((1, 0, 2))
        return torch.from_numpy(data)


class Pad:
    def __init__(self, num_frames: int = 150) -> None:
        self.num_frames = num_frames

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        padded_data = torch.zeros((self.num_frames, data.shape[1], data.shape[2]))
        L = data.shape[0]
        if L < self.num_frames:
            padded_data[:L, :, :] = data
            rest = self.num_frames - L
            num = int(np.ceil(rest / L))
            pad = torch.concatenate([data for _ in range(num)], 0)[:rest]
            padded_data[L:, :, :] = pad
        else:
            padded_data[:, :, :] = data[:self.num_frames, :, :]
        return padded_data


class Shift:
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        Move the landmark position interval to improve performance
        """
        return data - 0.5


class TensorToDict:
    def __call__(self, data: np.ndarray) -> dict:
        """
        Converts the tensor representation of the pose data into a dictionary.

        :param data: np.ndarray containing the pose data
        :return: Dictionary with the pose data
        """
        data_array = data.numpy()
        output = {}
        for idx, landmark in enumerate(LANDMARKS):
            output[landmark] = data_array[:, idx]
        return output


class DictToTensor:
    def __call__(self, data: dict) -> np.ndarray:
        """
        Converts the dictionary representation of the pose data into a tensor.

        :param data: Dictionary containing the pose data
        :return: np.ndarray with the pose data
        """
        output = np.empty((len(data["leftEar"]), len(LANDMARKS), 2))
        for idx, landmark in enumerate(LANDMARKS):
            output[:, idx, 0] = [frame[0] for frame in data[landmark]]
            output[:, idx, 1] = [frame[1] for frame in data[landmark]]
        return torch.from_numpy(output)


class SingleBodyDictNormalize:
    def __call__(self, row: dict) -> dict:
        """
        Normalizes the skeletal data for a given sequence of frames with signer's body pose data. The normalization follows
        the definition from our paper.

        :param row: Dictionary containing key-value pairs with joint identifiers and corresponding lists (sequences) of
                    that particular joints coordinates
        :return: Dictionary with normalized skeletal data (following the same schema as input data)
        """
        sequence_size = len(row["leftEar"])
        valid_sequence = True
        original_row = row

        last_starting_point, last_ending_point = None, None

        # Treat each element of the sequence (analyzed frame) individually
        for sequence_index in range(sequence_size):
            # Prevent from even starting the analysis if some necessary elements are not present
            if (
                row["leftShoulder"][sequence_index][0] == 0
                or row["rightShoulder"][sequence_index][0] == 0
            ) and (
                row["neck"][sequence_index][0] == 0 or row["nose"][sequence_index][0] == 0
            ):
                if not last_starting_point:
                    valid_sequence = False
                    continue

                else:
                    starting_point, ending_point = last_starting_point, last_ending_point

            else:

                # NOTE:
                #
                # While in the paper, it is written that the head metric is calculated by halving the shoulder distance,
                # this is meant for the distance between the very ends of one's shoulder, as literature studying body
                # metrics and ratios generally states. The Vision Pose Estimation API, however, seems to be predicting
                # rather the center of one's shoulder. Based on our experiments and manual reviews of the data, employing
                # this as just the plain shoulder distance seems to be more corresponding to the desired metric.
                #
                # Please, review this if using other third-party pose estimation libraries.

                if (
                    row["leftShoulder"][sequence_index][0] != 0
                    and row["rightShoulder"][sequence_index][0] != 0
                ):
                    left_shoulder = (
                        row["leftShoulder"][sequence_index][0],
                        row["leftShoulder"][sequence_index][1],
                    )
                    right_shoulder = (
                        row["rightShoulder"][sequence_index][0],
                        row["rightShoulder"][sequence_index][1],
                    )
                    shoulder_distance = (
                        ((left_shoulder[0] - right_shoulder[0]) ** 2)
                        + ((left_shoulder[1] - right_shoulder[1]) ** 2)
                    ) ** 0.5
                    head_metric = shoulder_distance
                else:
                    neck = (row["neck"][sequence_index][0], row["neck"][sequence_index][1])
                    nose = (row["nose"][sequence_index][0], row["nose"][sequence_index][1])
                    neck_nose_distance = (
                        ((neck[0] - nose[0]) ** 2) + ((neck[1] - nose[1]) ** 2)
                    ) ** 0.5
                    head_metric = neck_nose_distance

                # Set the starting and ending point of the normalization bounding box
                # starting_point = [row["neck"][sequence_index][0] - 3 * head_metric,
                #                  row["leftEye"][sequence_index][1] + (head_metric / 2)]
                starting_point = [
                    row["neck"][sequence_index][0] - 3 * head_metric,
                    row["leftEye"][sequence_index][1] + head_metric,
                ]
                ending_point = [
                    row["neck"][sequence_index][0] + 3 * head_metric,
                    starting_point[1] - 6 * head_metric,
                ]

                last_starting_point, last_ending_point = starting_point, ending_point

            # Ensure that all of the bounding-box-defining coordinates are not out of the picture
            if starting_point[0] < 0:
                starting_point[0] = 0
            if starting_point[1] < 0:
                starting_point[1] = 0
            if ending_point[0] < 0:
                ending_point[0] = 0
            if ending_point[1] < 0:
                ending_point[1] = 0

            # Normalize individual landmarks and save the results
            for identifier in BODY_LANDMARKS:
                key = identifier

                # Prevent from trying to normalize incorrectly captured points
                if row[key][sequence_index][0] == 0:
                    continue

                if any(
                    [
                        (ending_point[0] - starting_point[0]) == 0,
                        (starting_point[1] - ending_point[1]) == 0,
                    ]
                ):
                    logging.info(f"Problematic normalization with {key}")
                    valid_sequence = False
                    break

                normalized_x = (row[key][sequence_index][0] - starting_point[0]) / (
                    ending_point[0] - starting_point[0]
                )
                normalized_y = (row[key][sequence_index][1] - ending_point[1]) / (
                    starting_point[1] - ending_point[1]
                )

                row[key][sequence_index] = list(row[key][sequence_index])

                row[key][sequence_index][0] = normalized_x
                row[key][sequence_index][1] = normalized_y

        if valid_sequence:
            return row
        return original_row


class SPOTERSingleHandDictNormalize:
    def __call__(self, row: dict) -> dict:
        """
        Normalizes the skeletal data for a given sequence of frames with signer's hand pose data. The normalization follows
        the definition from our paper.

        :param row: Dictionary containing key-value pairs with joint identifiers and corresponding lists (sequences) of
                    that particular joints coordinates
        :return: Dictionary with normalized skeletal data (following the same schema as input data)
        """

        hand_landmarks = {0: [], 1: []}

        # Determine how many hands are present in the dataset
        range_hand_size = 1
        if "wrist_1" in row.keys():
            range_hand_size = 2

        # Construct the relevant identifiers
        for identifier in HAND_LANDMARKS:
            for hand_index in range(range_hand_size):
                hand_landmarks[hand_index].append(identifier + "_" + str(hand_index))

        # Treat each hand individually
        for hand_index in range(range_hand_size):

            sequence_size = len(row["wrist_" + str(hand_index)])

            # Treat each element of the sequence (analyzed frame) individually
            for sequence_index in range(sequence_size):

                # Retrieve all of the X and Y values of the current frame
                landmarks_x_values = [
                    row[key][sequence_index][0]
                    for key in hand_landmarks[hand_index]
                    if row[key][sequence_index][0] != 0
                ]
                landmarks_y_values = [
                    row[key][sequence_index][1]
                    for key in hand_landmarks[hand_index]
                    if row[key][sequence_index][1] != 0
                ]

                # Prevent from even starting the analysis if some necessary elements are not present
                if not landmarks_x_values or not landmarks_y_values:
                    continue

                # Calculate the deltas
                width, height = max(landmarks_x_values) - min(landmarks_x_values), max(
                    landmarks_y_values
                ) - min(landmarks_y_values)
                if width > height:
                    delta_x = 0.1 * width
                    delta_y = delta_x + ((width - height) / 2)
                else:
                    delta_y = 0.1 * height
                    delta_x = delta_y + ((height - width) / 2)

                # Set the starting and ending point of the normalization bounding box
                starting_point = (
                    min(landmarks_x_values) - delta_x,
                    min(landmarks_y_values) - delta_y,
                )
                ending_point = (
                    max(landmarks_x_values) + delta_x,
                    max(landmarks_y_values) + delta_y,
                )

                # Normalize individual landmarks and save the results
                for identifier in HAND_LANDMARKS:
                    key = identifier + "_" + str(hand_index)

                    # Prevent from trying to normalize incorrectly captured points
                    if (
                        row[key][sequence_index][0] == 0
                        or (ending_point[0] - starting_point[0]) == 0
                        or (starting_point[1] - ending_point[1]) == 0
                    ):
                        continue

                    normalized_x = (row[key][sequence_index][0] - starting_point[0]) / (
                        ending_point[0] - starting_point[0]
                    )
                    normalized_y = (row[key][sequence_index][1] - starting_point[1]) / (
                        ending_point[1] - starting_point[1]
                    )

                    row[key][sequence_index] = list(row[key][sequence_index])

                    row[key][sequence_index][0] = normalized_x
                    row[key][sequence_index][1] = normalized_y

        return row
