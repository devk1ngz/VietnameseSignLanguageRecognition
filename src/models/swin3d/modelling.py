import torch
import torch.nn as nn
from .configuration import Swin3DConfig
from transformers import ImageProcessingMixin, PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput


class Swin3DImageProcessor(ImageProcessingMixin):
    def __init__(self, config: Swin3DConfig = Swin3DConfig(), **kwargs) -> None:
        super().__init__(**kwargs)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.size = {"height": 224, "width": 224}
        self.min_resize_size = 256
        self.max_resize_size = 320
        self.num_frames = config.num_frames


class Swin3DForVideoClassification(PreTrainedModel):
    config_class = Swin3DConfig

    def __init__(
        self,
        config: Swin3DConfig = Swin3DConfig(),
        label2id: dict = None,
        id2label: dict = None,
    ) -> None:
        super().__init__(config=config)
        self.label2id = label2id if label2id is not None else config.label2id
        self.id2label = id2label if id2label is not None else config.id2label
        self.num_classes = len(self.label2id)

        weights = config.pretrained if config.pretrained != "DEFAULT" else "DEFAULT"
        self.model = torch.hub.load(
            "facebookresearch/pytorchvideo",
            config.arch,
            pretrained=(weights == "DEFAULT"),
        )

        # Replace the classification head
        if hasattr(self.model, "head"):
            in_features = self.model.head.in_features
            self.model.head = nn.Linear(in_features, self.num_classes)
        elif hasattr(self.model, "fc"):
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
