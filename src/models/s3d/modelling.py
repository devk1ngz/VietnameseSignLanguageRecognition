import torch
import torch.nn as nn
from .configuration import S3DConfig
from transformers import ImageProcessingMixin, PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput
from torchvision.models.video import s3d, S3D_Weights


class S3DImageProcessor(ImageProcessingMixin):
    def __init__(self, config: S3DConfig = S3DConfig(), **kwargs) -> None:
        super().__init__(**kwargs)
        self.mean = [0.43216, 0.394666, 0.37645]
        self.std = [0.22803, 0.22145, 0.216989]
        self.size = {"height": 224, "width": 224}
        self.min_resize_size = 256
        self.max_resize_size = 320
        self.num_frames = config.num_frames


class S3DForVideoClassification(PreTrainedModel):
    config_class = S3DConfig

    def __init__(
        self,
        config: S3DConfig = S3DConfig(),
        label2id: dict = None,
        id2label: dict = None,
    ) -> None:
        super().__init__(config=config)
        self.label2id = label2id if label2id is not None else config.label2id
        self.id2label = id2label if id2label is not None else config.id2label
        self.num_classes = len(self.label2id)

        if config.pretrained == "DEFAULT":
            self.model = s3d(weights=S3D_Weights.DEFAULT)
        else:
            self.model = s3d(weights=None)

        # Replace the classification head
        # S3D classifier is a Sequential with Conv3d as the last layer
        in_channels = self.model.classifier[1].in_channels
        self.model.classifier[1] = nn.Conv3d(
            in_channels, self.num_classes, kernel_size=1, stride=1
        )

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
        # S3D output is (batch, num_classes, 1, 1, 1), squeeze spatial dims
        if logits.dim() == 5:
            logits = logits.squeeze(-1).squeeze(-1).squeeze(-1)
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return ImageClassifierOutput(loss=loss, logits=logits)
        return ImageClassifierOutput(logits=logits)
