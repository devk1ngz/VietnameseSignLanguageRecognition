import torch
from .configuration import VideoMAEConfig
from transformers import ImageProcessingMixin, PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput
from transformers import VideoMAEImageProcessor as HFVideoMAEImageProcessor
from transformers import VideoMAEForVideoClassification as HFVideoMAEForVideoClassification


class VideoMAEImageProcessor(ImageProcessingMixin):
    def __init__(self, config: VideoMAEConfig = VideoMAEConfig(), **kwargs) -> None:
        super().__init__(**kwargs)
        processor = HFVideoMAEImageProcessor.from_pretrained(config.pretrained)
        self.mean = processor.image_mean
        self.std = processor.image_std
        if "shortest_edge" in processor.size:
            height = width = processor.size["shortest_edge"]
        else:
            height = processor.size["height"]
            width = processor.size["width"]
        self.size = {
            "height": height,
            "width": width,
        }

        self.min_resize_size = 256
        self.max_resize_size = 320

        self.num_frames = config.num_frames


class VideoMAEForVideoClassification(PreTrainedModel):
    def __init__(
        self,
        config: VideoMAEConfig = VideoMAEConfig(),
        label2id: dict = None,
        id2label: dict = None,
    ) -> None:
        super().__init__(config=config)
        self.label2id = label2id if label2id is not None else config.label2id
        self.id2label = id2label if id2label is not None else config.id2label
        self.model = HFVideoMAEForVideoClassification.from_pretrained(
            config.pretrained,
            label2id=self.label2id,
            id2label=self.id2label,
            ignore_mismatched_sizes=True,
        )
        self.num_classes = len(self.label2id)

        for i, param in enumerate(self.model.parameters()):
            if i >= config.num_frozen_layers:
                break
            param.requires_grad = False

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> torch.Tensor:
        logits = self.model(pixel_values).logits
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return ImageClassifierOutput(loss=loss, logits=logits)
        return ImageClassifierOutput(logits=logits)
