import torch
import torch.nn as nn
from .configuration import MViTConfig
from transformers import ImageProcessingMixin, PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput
from torchvision.models.video import (
    mvit_v1_b, MViT_V1_B_Weights,
    mvit_v2_s, MViT_V2_S_Weights,
)


_MODEL_REGISTRY = {
    "mvit_v1_b": (mvit_v1_b, MViT_V1_B_Weights),
    "mvit_v2_s": (mvit_v2_s, MViT_V2_S_Weights),
}


class MViTImageProcessor(ImageProcessingMixin):
    def __init__(self, config: MViTConfig = MViTConfig(), **kwargs) -> None:
        super().__init__(**kwargs)
        self.mean = [0.45, 0.45, 0.45]
        self.std = [0.225, 0.225, 0.225]
        self.size = {"height": 224, "width": 224}
        self.min_resize_size = 256
        self.max_resize_size = 320
        self.num_frames = config.num_frames


class MViTForVideoClassification(PreTrainedModel):
    config_class = MViTConfig

    def __init__(
        self,
        config: MViTConfig = MViTConfig(),
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
        in_features = self.model.head[1].in_features
        self.model.head[1] = nn.Linear(in_features, self.num_classes)

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
