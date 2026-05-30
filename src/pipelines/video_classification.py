import torch
import numpy as np
from typing import Union
import onnxruntime as ort
from transformers import Pipeline, AutoConfig
from huggingface_hub import hf_hub_download
from pytorchvideo.transforms import Normalize
from torchvision.transforms.v2 import (
    Compose,
    Lambda,
    Resize,
    CenterCrop,
)


class VideoClassificationPipeline(Pipeline):
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

        mean = self.image_processor.mean
        std = self.image_processor.std
        min_resize_size = self.image_processor.min_resize_size
        crop_size = (
            self.image_processor.size["height"],
            self.image_processor.size["width"]
        )
        self.transforms = Compose(
            [
                Lambda(lambda x: x / 255.0),
                Normalize(mean, std),
                Resize(min_resize_size),
                CenterCrop(crop_size),
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

        # Uniformly sample the frames
        if inputs.size(0) > self.image_processor.num_frames:
            idxs = np.linspace(
                start=0,
                stop=len(inputs) - 1,
                num=self.image_processor.num_frames,
                dtype=int,
            )
            inputs = inputs[idxs]
        # Permute to (channels, time, height, width)
        inputs = inputs.permute(3, 0, 1, 2)
        # Transform the inputs
        inputs = self.transforms(inputs)
        # Permute to (batch, channels, time, height, width)
        inputs = inputs.permute(1, 0, 2, 3).unsqueeze(0)

        return inputs

    def _forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if isinstance(self.model, ort.InferenceSession):
            inputs = inputs.cpu().numpy()
            return torch.from_numpy(self.model.run(None, {"pixel_values": inputs})[0])
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
