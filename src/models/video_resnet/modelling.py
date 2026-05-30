import torch
import torch.nn as nn
from .configuration import VideoResNetConfig
from transformers import ImageProcessingMixin, PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput
from torchvision.models.video import (
    r3d_18, R3D_18_Weights,
    mc3_18, MC3_18_Weights,
    r2plus1d_18, R2Plus1D_18_Weights,
)


_MODEL_REGISTRY = {
    "r3d_18": (r3d_18, R3D_18_Weights),
    "mc3_18": (mc3_18, MC3_18_Weights),
    "r2plus1d_18": (r2plus1d_18, R2Plus1D_18_Weights),
}


class VideoResNetImageProcessor(ImageProcessingMixin):
    def __init__(self, config: VideoResNetConfig = VideoResNetConfig(), **kwargs) -> None:
        super().__init__(**kwargs)
        self.mean = [0.43216, 0.394666, 0.37645]
        self.std = [0.22803, 0.22145, 0.216989]
        self.size = {"height": 112, "width": 112}
        self.min_resize_size = 128
        self.max_resize_size = 171
        self.num_frames = config.num_frames


class VideoResNetForVideoClassification(PreTrainedModel):
    config_class = VideoResNetConfig

    def __init__(
        self,
        config: VideoResNetConfig = VideoResNetConfig(),
        label2id: dict = None,
        id2label: dict = None,
    ) -> None:
        super().__init__(config=config)
        self.label2id = label2id if label2id is not None else config.label2id
        self.id2label = id2label if id2label is not None else config.id2label
        self.num_classes = len(self.label2id)

        model_fn, weights_cls = _MODEL_REGISTRY[config.arch]
        if config.pretrained == "DEFAULT":
            self.model = model_fn(weights=weights_cls.DEFAULT)
        else:
            self.model = model_fn(weights=None)

        # Replace the classification head
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, self.num_classes)

        # Freeze layers
        for i, param in enumerate(self.model.parameters()):
            if i >= config.num_frozen_layers:
                break
            param.requires_grad = False

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> torch.Tensor:
        logits = self.model(pixel_values)
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return ImageClassifierOutput(loss=loss, logits=logits)
        return ImageClassifierOutput(logits=logits)
